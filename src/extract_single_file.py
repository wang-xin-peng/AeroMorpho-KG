"""
单文件知识三元组抽取脚本
功能：
1. 读取指定的Markdown文件
2. 使用OneKE模型根据预定义的Schema抽取实体关系三元组
3. 去重后输出为JSONL格式文件
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from extract_triples import Extractor, split_text
from common import dump_jsonl
from schema import Schema


def run_extract_single_file(
    input_file: str,
    output_dir: str,
    schema_path: str,
    model_path: str = "model/OneKE",
    load_in_4bit: bool = True,
    chunk_chars: int = 150,
    overlap: int = 30,
) -> int:
    """
    单文件抽取主函数
    流程：
    1. 加载Schema配置
    2. 初始化OneKE抽取器
    3. 读取指定的Markdown文件
    4. 切分文本块并抽取
    5. 去重后保存结果
    Args:
        input_file: 输入Markdown文件路径
        output_dir: 输出目录路径
        schema_path: Schema配置文件路径
        model_path: OneKE模型路径
        load_in_4bit: 是否使用4bit量化
        chunk_chars: 文本块最大字符数
        overlap: 重叠字符数
    Returns:
        抽取的三元组总数
    """
    print("="*60)
    print("单文件抽取三元组")
    print("="*60)

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: 加载Schema配置
    print(f"\n[Step 1/3] 加载 Schema 配置: {schema_path}", flush=True)
    schema = Schema.from_json(schema_path)
    relation_count = len(schema.get_relation_names())
    print(f"[Schema] 加载了 {relation_count} 个关系类型", flush=True)
    
    # Step 2: 初始化抽取器
    print(f"\n[Step 2/3] 初始化 OneKE 抽取器", flush=True)
    extractor = Extractor(model_path=model_path, load_in_4bit=load_in_4bit)

    all_rows: List[Dict] = []
    dedup = set() 
    
    # Step 3: 读取文件并抽取
    print(f"\n[Step 3/3] 开始抽取三元组", flush=True)
    print(f"[文件] {input_path.name}")
    print(f"[分块配置] chunk_chars={chunk_chars}, overlap={overlap}, 每个关系批次={extractor.split_num}个", flush=True)
    
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    chunks = split_text(text, max_chars=chunk_chars, overlap=overlap)
    
    # 计算分块统计信息
    avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
    print(f"\n{'='*60}")
    print(f"[文件] {input_path.name}")
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
                    "source": input_path.name,      # 记录来源文件
                    "method": "oneke_deepke_pipeline",  # 记录抽取方法
                }
            )
            doc_count += 1
    
    print(f"\n[完成] {input_path.name}: 总计 {doc_count} 个唯一三元组\n", flush=True)

    # Step 4: 保存结果
    output_file = output_path / "triples_final.jsonl"
    dump_jsonl(output_file, all_rows)
    print(f"[DONE] OneKE triples = {len(all_rows)}, saved to {output_file}")
    return len(all_rows)


def main() -> None:
    """
    使用示例：
    python extract_single_file.py \
        --input-file tmpp/tmp/1.md \
        --output-dir tmpp/output
    """

    parser = argparse.ArgumentParser(description="单文件知识三元组抽取")
    parser.add_argument("--input-file", required=True, help="输入Markdown文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录路径")
    parser.add_argument("--schema-path", default="./config/relation_types.json", help="关系类型配置文件路径")
    parser.add_argument("--model-path", default="model/OneKE", help="OneKE模型路径")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="使用4bit量化")
    parser.add_argument("--no-quantize", dest="load_in_4bit", action="store_false", help="不使用量化")
    parser.add_argument("--chunk-chars", type=int, default=150, help="文本块最大字符数")
    parser.add_argument("--overlap", type=int, default=30, help="重叠字符数")
    args = parser.parse_args()

    run_extract_single_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        schema_path=args.schema_path,
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
