"""
串联PDF解析、OneKE抽取和Yuan-Embedding融合的端到端流水线。
"""

import argparse
import asyncio

from extract_triples import run_extract
from fuse_entities import run_fusion
from parse_docs import run_parse


def main() -> None:
    """
    流水线主函数
    执行顺序：
    1. 解析PDF为Markdown
    2. 用OneKE抽取三元组
    3. 用向量模型融合归一化
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="./变构飞行器")
    parser.add_argument("--parse-dir", default="./data/parsed")
    parser.add_argument("--raw-triples", default="./data/triples_raw/triples.jsonl")
    parser.add_argument("--fused-triples", default="./data/triples_fused/triples_fused.jsonl")
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析层")
    parser.add_argument("--schema-path", default="config/schema.json")
    parser.add_argument("--oneke-model", default="model/OneKE")
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--embedding-model", default="model/Yuan-Embedding")
    parser.add_argument("--entity-threshold", type=float, default=0.85)
    parser.add_argument("--relation-threshold", type=float, default=0.9)
    args = parser.parse_args()

    # 解析PDF为Markdown
    if not args.skip_parse:
        asyncio.run(run_parse(args.pdf_dir, args.parse_dir, skip_exists=True))
    # 抽取三元组
    raw_count = run_extract(
        parsed_dir=args.parse_dir,
        out_jsonl=args.raw_triples,
        schema_path=args.schema_path,
        model_path=args.oneke_model,
        chunk_chars=args.chunk_chars,
    )
    # 知识融合
    fused_count = run_fusion(
        in_jsonl=args.raw_triples,
        out_jsonl=args.fused_triples,
        model_name=args.embedding_model,  # argparse会把连字符转为下划线
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
    )
    print(f"[PIPELINE DONE] raw={raw_count}, fused={fused_count}")


if __name__ == "__main__":
    main()