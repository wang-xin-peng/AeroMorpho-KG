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


def split_text(text: str, max_chars: int = 1200) -> List[str]:
    """
    将长文本按句子边界切分成多个文本块(OneKE模型有输入长度限制)
    Args:
        text: 原始文本
        max_chars: 每个文本块的最大字符数，默认1200字符
    Returns:
        切分后的文本块列表
    """

    parts = re.split(r"(?<=[。！？\n])", text)
    chunks: List[str] = []
    cur = ""
    
    for p in parts:
        if len(cur) + len(p) <= max_chars:
            cur += p
        else:
            if cur.strip():
                chunks.append(cur.strip())
            cur = p
    
    if cur.strip():
        chunks.append(cur.strip())
    
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
    
    def __init__(
        self,
        model_path: str = "./model/OneKE",
        load_in_4bit: bool = True,
        max_new_tokens: int = 768,
        split_num: int = 4,  # RE任务的推荐切分长度是4
    ) -> None:
        """
        初始化抽取器，加载模型
        Args:
            model_path: 模型路径或HuggingFace模型名
            max_new_tokens: 生成的最大token数
            split_num: 每次轮询抽取的schema数量
        """
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 模型参数
        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
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
        
        # 对关系类型进行轮询抽取
        for i in range(0, len(relation_names), self.split_num):
            batch_relations = relation_names[i:i + self.split_num]
            
            # 生成 OneKE schema（自动选择基础模式或增强模式）
            oneke_schema = schema.get_relation_schema_dict(batch_relations)
            
            # 构建 OneKE 指令
            instruction = schema.instruction
            prompt = self._build_oneke_prompt(instruction, oneke_schema, chunk)
            
            # 执行推理
            triples = self._run_inference(prompt)
            all_triples.extend(triples)
        
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
        # 分词并移动到模型设备
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,      
                temperature=0.0,     
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型生成的部分（去掉输入提示词）
        tail_text = text[len(prompt):] if text.startswith(prompt) else text
        
        # 解析JSON数组
        return parse_json_array(tail_text)


def run_extract(
    parsed_dir: str,
    out_jsonl: str,
    schema_path: str,
    model_path: str = "./model/OneKE",
    load_in_4bit: bool = True,
    chunk_chars: int = 1200,
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
        chunk_chars: 文本块最大字符数
    Returns:
        抽取的三元组总数
    """

    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        raise FileNotFoundError(f"解析目录不存在: {parsed_dir}")

    # Step 1: 加载DeepKE Schema配置
    schema = DeepKESchema.from_json(schema_path)
    
    # Step 2: 初始化OneKE抽取器
    extractor = Extractor(model_path=model_path, load_in_4bit=load_in_4bit)

    all_rows: List[Dict] = []
    dedup = set() 
    
    # Step 3: 遍历所有Markdown文件
    md_files = sorted(parsed_path.glob("*.md"))
    
    # Step 4: 对每个文件切分文本块并抽取
    for md_file in tqdm(md_files, desc="OneKE Extracting"):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        chunks = split_text(text, max_chars=chunk_chars)
        
        doc_count = 0
        for chunk in chunks:
            rows = extractor.infer_chunk(schema, chunk)
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
        
        print(f"[EXTRACT] {md_file.name}: {doc_count} triples")

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
        --schema-path ./config/schema.json
    """

    parser = argparse.ArgumentParser(description="使用OneKE抽取知识三元组")
    parser.add_argument("--parsed-dir", required=True, help="LlamaParse解析后的Markdown目录")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径")
    parser.add_argument("--schema-path", default="./config/schema.json", help="Schema配置文件路径")
    parser.add_argument("--model-path", default="./model/OneKE", help="OneKE模型路径")
    parser.add_argument("--chunk-chars", type=int, default=1200, help="文本块最大字符数")
    args = parser.parse_args()

    run_extract(
        parsed_dir=args.parsed_dir,
        out_jsonl=args.out_jsonl,
        schema_path=args.schema_path,
        model_path=args.model_path,
        chunk_chars=args.chunk_chars,
    )


if __name__ == "__main__":
    main()