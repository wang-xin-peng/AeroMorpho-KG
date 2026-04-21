"""
基于DeepKE Schema与OneKE模型抽取知识三元组
功能：
1. 读取LlamaParse解析后的Markdown文件
2. 使用OneKE大模型根据预定义的Schema抽取实体关系三元组
3. 去重后输出为JSONL格式文件
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Union

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import dump_jsonl
from schema import DeepKESchema


def split_text(text: str, max_chars: int = 150, overlap: int = 30) -> List[str]:
    """
    将长文本按句子边界切分成多个文本块，使用重叠窗口避免边界处关系丢失
    
    改进策略：
    1. 先按句子切分（使用中文标点：。！？；）
    2. 将句子组合成不超过max_chars的文本块
    3. 使用overlap字符的重叠窗口，避免跨块关系丢失
    
    Args:
        text: 原始文本
        max_chars: 每个文本块的最大字符数，默认150字符（官方配置cutoff_len = 512tokens, prompt
        剩余给input的tokens约250个，一个汉字1.5~2个token，这里保守选择150）
        overlap: 重叠字符数，默认30字符（20%重叠）
    Returns:
        切分后的文本块列表
    """
    # 按句子切分（保留句子完整性）
    # 使用正向预查，保留分隔符
    sentences = re.split(r'(?<=[。！？；])', text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前句子本身就超过max_chars，单独作为一个chunk
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence.strip())
            continue
        
        # 如果加上当前句子不超过max_chars，添加到当前chunk
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            # 当前chunk已满，保存并开始新chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                
                # 创建重叠：从当前chunk末尾取overlap个字符作为新chunk的开头
                if len(current_chunk) > overlap:
                    # 找到overlap范围内最近的句子边界
                    overlap_text = current_chunk[-overlap:]
                    # 尝试从完整句子开始
                    sentence_start = max(
                        overlap_text.rfind('。') + 1,
                        overlap_text.rfind('！') + 1,
                        overlap_text.rfind('？') + 1,
                        overlap_text.rfind('；') + 1,
                        0
                    )
                    if sentence_start > 0:
                        current_chunk = overlap_text[sentence_start:] + sentence
                    else:
                        current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = sentence
    
    # 添加最后一个chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def parse_json_array(text: str) -> List[Dict]:
    """
    从模型输出文本中解析JSON格式的三元组
    支持两种格式：
    1. OneKE格式：{"关系名": [{"subject": "...", "object": "..."}]}
    2. 标准格式：[{"head": "...", "relation": "...", "tail": "..."}]
    
    Args:
        text: 模型输出的原始文本，可能包含额外的解释文字
    Returns:
        解析后的三元组列表，每个三元组包含head、relation、tail字段
    """
    # 使用正则表达式提取JSON部分
    match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if not match:
        return []
    
    raw = match.group(0)  
    
    try:
        data = json.loads(raw)
    except Exception:
        return []
    
    rows: List[Dict] = []
    
    # 格式1: OneKE格式 {"关系名": [{"subject": "...", "object": "..."}]}
    if isinstance(data, dict):
        for relation, items in data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        h = str(item.get("subject", "")).strip()
                        t = str(item.get("object", "")).strip()
                        if h and relation and t:
                            rows.append({"head": h, "relation": relation, "tail": t})
    
    # 格式2: 标准格式 [{"head": "...", "relation": "...", "tail": "..."}]
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            
            h = str(item.get("head", "")).strip()
            r = str(item.get("relation", "")).strip()
            t = str(item.get("tail", "")).strip()
            if h and r and t:
                rows.append({"head": h, "relation": r, "tail": t})
    
    return rows


class Extractor:
    """
    OneKE知识抽取器封装类
    功能：
    1. 加载OneKE预训练模型
    2. 接收Schema提示词和文本块
    3. 调用模型推理并解析结果
    4. 支持轮询方式抽取（每次抽取固定数量的schema）
    """
    
    # OneKE官方推荐的RE任务instruction
    RE_INSTRUCTION = "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。"
    
    def __init__(
        self,
        model_path: str = "model/OneKE",
        load_in_4bit: bool = False,  # 显存充裕，不用量化
        max_new_tokens: int = 300,  # 官方推荐
        split_num: int = 4,  # RE任务官方推荐值
    ) -> None:
        """
        初始化抽取器，加载模型
        Args:
            model_path: 模型路径或HuggingFace模型名
            max_new_tokens: 生成的最大token数（官方推荐300）
            split_num: 每次轮询抽取的schema数量（RE任务官方推荐4）
        """
        print(f"\n[模型加载] 正在加载 OneKE 模型: {model_path}", flush=True)
        
        # 加载分词器
        print("[模型加载] 加载分词器...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 模型参数（参考官方配置）
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,  # 使用bf16精度
        }
        
        # 加载模型（不使用量化）
        print("[模型加载] 加载模型权重（bf16精度，无量化）...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        print(f"[模型加载] ✓ 模型加载完成", flush=True)
        print(f"[模型配置] max_new_tokens={max_new_tokens}, split_num={split_num}", flush=True)
        
        self.max_new_tokens = max_new_tokens
        self.split_num = split_num

    def infer_chunk(self, schema: DeepKESchema, chunk: str) -> List[Dict]:
        """
        对单个文本块进行推理，抽取三元组
        使用OneKE轮询方式：将schema切分成多个小块，分别抽取后合并结果
        Args:
            schema: DeepKESchema 对象，包含关系类型定义
            chunk: 待抽取的文本块
        Returns:
            从该文本块中抽取的三元组列表
        """
        # 获取所有关系名称
        relation_names = schema.get_relation_names()
        
        # 使用轮询方式抽取
        all_triples = []
        
        # 对关系类型进行轮询抽取（每次处理split_num个关系）
        num_batches = (len(relation_names) + self.split_num - 1) // self.split_num
        
        for i in range(0, len(relation_names), self.split_num):
            batch_idx = i // self.split_num + 1
            batch_relations = relation_names[i:i + self.split_num]
            
            print(f"    [轮询 {batch_idx}/{num_batches}] 处理关系: {', '.join(batch_relations[:3])}{'...' if len(batch_relations) > 3 else ''}", flush=True)
            
            # 生成 OneKE schema（字典格式：{关系名: 描述}）
            oneke_schema = schema.get_relation_schema_dict(batch_relations)
            
            # 构建 OneKE 指令（使用类常量中的instruction）
            prompt = self._build_oneke_prompt(self.RE_INSTRUCTION, oneke_schema, chunk)
            
            # 执行推理
            triples = self._run_inference(prompt)
            all_triples.extend(triples)
            
            if triples:
                print(f"    [轮询 {batch_idx}/{num_batches}] 抽取到 {len(triples)} 个三元组", flush=True)
        
        return all_triples
    
    def _build_oneke_prompt(self, instruction: str, schema: Union[List[str], Dict[str, str]], input_text: str) -> str:
        """
        构建OneKE标准格式的指令
        使用LLaMA对话格式：[INST] <<SYS>>...<</SYS>> {JSON指令} [/INST]
        支持两种 schema 格式：
        1. 列表格式（基础模式）：["关系1", "关系2"]
        2. 字典格式（增强模式）：{"关系1": "描述1", "关系2": "描述2"}
        
        参考 OneKE 文档的快速运行示例
        """
        import json
        
        # 通用的 system prompt（固定）
        system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n'
        
        # 构建 JSON 指令字符串
        sintruct = json.dumps({
            "instruction": instruction,
            "schema": schema,
            "input": input_text
        }, ensure_ascii=False)
        
        # 组合成完整 prompt
        prompt = '[INST] ' + system_prompt + sintruct + ' [/INST]'
        
        return prompt
    
    def _run_inference(self, prompt: str) -> List[Dict]:
        """
        执行模型推理并解析结果
        Args:
            prompt: OneKE格式的提示词
        Returns:
            解析后的三元组列表
        """
        import sys
        
        # 分词并移动到模型设备
        print("      [推理] 分词中...", end='', flush=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        print(f" 输入tokens: {inputs['input_ids'].shape[1]}", flush=True)
        
        # 模型推理
        print("      [推理] 生成中...", end='', flush=True)
        sys.stdout.flush()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,      
                temperature=None,  # 修复：temperature=0.0 时应该设为 None
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        print(" 完成", flush=True)
        
        # 解码输出
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型生成的部分（去掉输入提示词）
        tail_text = text[len(prompt):] if text.startswith(prompt) else text
        
        # 解析JSON数组
        triples = parse_json_array(tail_text)
        
        return triples


def run_extract(
    parsed_dir: str,
    out_jsonl: str,
    schema_path: str,
    model_path: str = "model/OneKE",
    load_in_4bit: bool = False,  # 显存充裕，不用量化
    chunk_chars: int = 150,  # 极小chunk，最大化召回率
    overlap: int = 30,  # 20%重叠
) -> int:
    """
    OneKE抽取主函数
    流程：
    1. 加载DeepKE Schema配置
    2. 初始化OneKE抽取器
    3. 遍历所有Markdown文件
    4. 对每个文件切分文本块并抽取
    5. 去重后保存结果
    Args:
        parsed_dir: LlamaParse解析后的Markdown文件目录
        out_jsonl: 输出JSONL文件路径
        schema_path: Schema配置文件路径
        model_path: OneKE模型路径
        load_in_4bit: 是否使用4bit量化
        chunk_chars: 文本块最大字符数（150字符，极小chunk策略）
        overlap: 重叠字符数（20%重叠，最大化覆盖）
    Returns:
        抽取的三元组总数
    """

    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        raise FileNotFoundError(f"解析目录不存在: {parsed_dir}")

    # Step 1: 加载DeepKE Schema配置
    print(f"\n[Step 1/3] 加载 Schema 配置: {schema_path}", flush=True)
    schema = DeepKESchema.from_json(schema_path)
    relation_count = len(schema.get_relation_names())
    print(f"[Schema] 加载了 {relation_count} 个关系类型", flush=True)
    
    # Step 2: 初始化OneKE抽取器
    print(f"\n[Step 2/3] 初始化 OneKE 抽取器", flush=True)
    extractor = Extractor(model_path=model_path, load_in_4bit=load_in_4bit)

    all_rows: List[Dict] = []
    dedup = set() 
    
    # Step 3: 遍历所有Markdown文件
    md_files = sorted(parsed_path.glob("*.md"))
    print(f"\n[Step 3/3] 开始抽取三元组", flush=True)
    print(f"[文件统计] 共 {len(md_files)} 个文档待处理", flush=True)
    print(f"[分块配置] chunk_chars={chunk_chars}, overlap={overlap}, 每个关系批次={extractor.split_num}个", flush=True)
    
    # Step 4: 对每个文件切分文本块并抽取
    for md_file in tqdm(md_files, desc="OneKE Extracting"):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text(text, max_chars=chunk_chars, overlap=overlap)
        
        # 计算分块统计信息
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        print(f"\n{'='*60}")
        print(f"[文件] {md_file.name}")
        print(f"  文本长度: {len(text)} 字符")
        print(f"  切分块数: {len(chunks)} 个")
        print(f"  平均块大小: {avg_chunk_size:.0f} 字符")
        print(f"  块大小范围: {min(len(c) for c in chunks) if chunks else 0} - {max(len(c) for c in chunks) if chunks else 0} 字符")
        print(f"{'='*60}", flush=True)
        
        doc_count = 0
        for idx, chunk in enumerate(chunks):
            print(f"\n  [Chunk {idx+1}/{len(chunks)}] 长度: {len(chunk)} 字符", flush=True)
            
            rows = extractor.infer_chunk(schema, chunk)
            chunk_triples = len(rows)
            
            if chunk_triples > 0:
                print(f"  [Chunk {idx+1}/{len(chunks)}] ✓ 抽取到 {chunk_triples} 个三元组", flush=True)
            else:
                print(f"  [Chunk {idx+1}/{len(chunks)}] - 未抽取到三元组", flush=True)
            
            for r in rows:
                key = (r["head"], r["relation"], r["tail"])
                if key in dedup:
                    continue  # 重复的三元组跳过
                dedup.add(key)
                all_rows.append(
                    {
                        "head": r["head"],
                        "relation": r["relation"],
                        "tail": r["tail"],
                        "source": md_file.name,      # 记录来源文件
                        "method": "oneke_deepke_pipeline",  # 记录抽取方法
                    }
                )
                doc_count += 1
        
        print(f"\n[完成] {md_file.name}: 总计 {doc_count} 个唯一三元组\n", flush=True)

    # Step 5: 保存结果
    dump_jsonl(out_jsonl, all_rows)
    print(f"[DONE] OneKE triples = {len(all_rows)}, saved to {out_jsonl}")
    return len(all_rows)


def main() -> None:
    """
    使用示例：
    python extract_triples.py \
        --parsed-dir ./ragtest/input \
        --out-jsonl ./triples.jsonl \
        --schema-path ./config/relation_types.json
    """

    parser = argparse.ArgumentParser(description="使用OneKE抽取知识三元组")
    parser.add_argument("--parsed-dir", required=True, help="LlamaParse解析后的Markdown目录")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径")
    parser.add_argument("--schema-path", default="./config/relation_types.json", help="关系类型配置文件路径")
    parser.add_argument("--model-path", default="model/OneKE", help="OneKE模型路径")
    parser.add_argument("--chunk-chars", type=int, default=150, help="文本块最大字符数（极小chunk策略）")
    parser.add_argument("--overlap", type=int, default=30, help="重叠字符数（20%重叠）")
    args = parser.parse_args()

    run_extract(
        parsed_dir=args.parsed_dir,
        out_jsonl=args.out_jsonl,
        schema_path=args.schema_path,
        model_path=args.model_path,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()