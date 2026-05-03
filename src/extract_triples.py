"""
使用DeepSeek API抽取知识三元组
功能：
1. 读取LlamaParse解析后的Markdown文件
2. 使用DeepSeek API根据预定义的Schema抽取实体关系三元组
3. 去重后输出为JSONL格式文件
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from common import dump_jsonl
from schema import Schema

load_dotenv()


def split_text(text: str, max_chars: int = 150, overlap: int = 30) -> List[str]:
    """
    将长文本按句子边界切分成多个文本块，使用重叠窗口避免边界处关系丢失
    功能：
    1. 先按句子切分（使用中文标点：。！？；）
    2. 将句子组合成不超过max_chars的文本块
    3. 使用overlap字符的重叠窗口，避免跨块关系丢失
    Args:
        text: 原始文本
        max_chars: 每个文本块的最大字符数，默认150字符
        overlap: 重叠字符数，默认30字符（20%重叠）
    Returns:
        切分后的文本块列表
    """

    # 按句子切分
    sentences = re.split(r'(?<=[。！？；])', text)
    
    # 过滤空句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前句子本身就超过max_chars，强制按max_chars切分
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # 将超长句子按max_chars强制切分
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars].strip())
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
    从模型输出文本中解析三元组
    {"关系名": [{"subject": "...", "object": "..."}]}
    Args:
        text: 模型输出的原始文本，可能包含额外的解释文字
    Returns:
        解析后的三元组列表，每个三元组包含head、relation、tail字段
    """

    rows: List[Dict] = []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for relation, items in data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            h = str(item.get("subject", "")).strip()
                            t = str(item.get("object", "")).strip()
                            if h and relation and t:
                                rows.append({"head": h, "relation": relation, "tail": t})
        return rows
    except json.JSONDecodeError:
        pass

    text = text.strip()
    start_idx = text.find('{')
    end_idx = text.rfind('}')

    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        print("[解析] 未找到JSON部分")
        return []

    raw = text[start_idx:end_idx + 1]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[解析] JSON解析失败: {e}")
        print(f"[原始响应] {text[:500]}")
        return []

    if isinstance(data, dict):
        for relation, items in data.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        h = str(item.get("subject", "")).strip()
                        t = str(item.get("object", "")).strip()
                        if h and relation and t:
                            rows.append({"head": h, "relation": relation, "tail": t})

    return rows


class Extractor:
    """
    DeepSeek API知识抽取器封装类
    功能：
    1. 连接DeepSeek API
    2. 接收Schema提示词和文本块
    3. 调用API推理并解析结果
    4. 轮询方式抽取schema
    """

    RE_INSTRUCTION = "你是专门进行关系抽取的专家。请从input中抽取出符合schema定义的关系三元组，不存在的关系返回空列表。请按照JSON字符串的格式回答。"

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "deepseek-ai/deepseek-v4-pro",
        max_tokens: int = 300,
        split_num: int = 4,
    ) -> None:

        if api_key is None:
            api_key = os.getenv("NVAPI_KEY")

        print(f"\n[API初始化] 正在连接 DeepSeek API: {model}", flush=True)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.split_num = split_num

        print(f"[API配置] model={model}, max_tokens={max_tokens}, split_num={split_num}", flush=True)

    def infer_chunk(self, schema: Schema, chunk: str) -> List[Dict]:
        """
        对单个文本块进行推理，抽取三元组
        使用轮询方式：将schema切分成多个小块，分别抽取后合并结果
        Args:
            schema: Schema 对象，包含关系类型定义
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
            
            # {关系名: 描述}
            oneke_schema = schema.get_relation_schema_dict(batch_relations)
            
            # 构建指令（使用类常量中的instruction）
            prompt = self._build_prompt(self.RE_INSTRUCTION, oneke_schema, chunk)
            
            # 执行推理
            triples = self._run_inference(prompt)
            all_triples.extend(triples)
            
            if triples:
                print(f"    [轮询 {batch_idx}/{num_batches}] 抽取到 {len(triples)} 个三元组", flush=True)
        
        return all_triples
    
    def _build_prompt(self, instruction: str, schema: Union[List[str], Dict[str, str]], input_text: str) -> str:
        """
        构建prompt
        [INST] <<SYS>>...<</SYS>> {JSON指令} [/INST]
        支持两种 schema 格式：
        1. 列表格式（基础模式）：["关系1", "关系2"]
        2. 字典格式（增强模式）：{"关系1": "描述1", "关系2": "描述2"}
        """
        
        # system prompt
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
        执行API推理并解析结果
        Args:
            prompt: OneKE格式的提示词
        Returns:
            解析后的三元组列表
        """

        print("      [推理] 调用API中...", end='', flush=True)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=0,
        )

        print(" 完成", flush=True)

        text = response.choices[0].message.content

        triples = parse_json_array(text)

        return triples


def run_extract(
    in_dir: str,
    out_jsonl: str,
    schema_path: str,
    api_key: str = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    model: str = "deepseek-ai/deepseek-v4-pro",
    chunk_chars: int = 300,
    overlap: int = 30,
) -> int:
    """
    DeepSeek API抽取主函数
    流程：
    1. 加载Schema配置
    2. 初始化DeepSeek API抽取器
    3. 遍历所有Markdown文件
    4. 对每个文件切分文本块并抽取
    5. 去重后保存结果
    Args:
        in_dir: 预处理后的Markdown文件目录
        out_jsonl: 输出JSONL文件路径
        schema_path: Schema配置文件路径
        api_key: API密钥
        base_url: API基础URL
        model: 模型名称
        chunk_chars: 文本块最大字符数
        overlap: 重叠字符数
    Returns:
        抽取的三元组总数
    """
    print("="*60)
    print("抽取三元组")
    print("="*60)

    parsed_path = Path(in_dir)
    if not parsed_path.exists():
        raise FileNotFoundError(f"解析目录不存在: {in_dir}")

    print(f"\n[Step 1/3] 加载 Schema 配置: {schema_path}", flush=True)
    schema = Schema.from_json(schema_path)
    relation_count = len(schema.get_relation_names())
    print(f"[Schema] 加载了 {relation_count} 个关系类型", flush=True)

    print(f"\n[Step 2/3] 初始化 DeepSeek API 抽取器", flush=True)
    extractor = Extractor(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    all_rows: List[Dict] = []
    dedup = set() 
    
    # Step 3: 遍历所有Markdown文件
    md_files = sorted(parsed_path.glob("*.md"))
    print(f"\n[Step 3/3] 开始抽取三元组", flush=True)
    print(f"[文件统计] 共 {len(md_files)} 个文档待处理", flush=True)
    print(f"[分块配置] chunk_chars={chunk_chars}, overlap={overlap}, 每个关系批次={extractor.split_num}个", flush=True)
    
    # Step 4: 对每个文件切分文本块并抽取
    for md_file in tqdm(md_files, desc="DeepSeek Extracting"):
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
                        "method": "deepseek_api",  # 记录抽取方法
                    }
                )
                doc_count += 1
        
        print(f"\n[完成] {md_file.name}: 总计 {doc_count} 个唯一三元组\n", flush=True)

    # Step 5: 保存结果
    dump_jsonl(out_jsonl, all_rows)
    print(f"[DONE] DeepSeek triples = {len(all_rows)}, saved to {out_jsonl}")
    return len(all_rows)


def main() -> None:
    """
    使用示例：
    python extract_triples.py \
        --in-dir ./test/input \
        --out-jsonl ./test.jsonl \
    """

    parser = argparse.ArgumentParser(description="使用DeepSeek API抽取知识三元组")
    parser.add_argument("--in-dir", required=True, help="预处理后的Markdown目录")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径")
    parser.add_argument("--schema-path", default="./config/relation_types.json", help="关系类型配置文件路径")
    parser.add_argument("--api-key", default=None, help="API密钥")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1", help="API基础URL")
    parser.add_argument("--model", default="deepseek-ai/deepseek-v4-pro", help="模型名称")
    parser.add_argument("--chunk-chars", type=int, default=300, help="文本块最大字符数")
    parser.add_argument("--overlap", type=int, default=30, help="重叠字符数")
    args = parser.parse_args()

    run_extract(
        in_dir=args.in_dir,
        out_jsonl=args.out_jsonl,
        schema_path=args.schema_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()