"""
串联PDF解析、文档预处理、OneKE抽取和Yuan-Embedding融合的端到端流水线。
"""

import argparse
import asyncio

from extract_triples import run_extract
from fuse_entities import run_fusion
from parse_docs import run_parse
from preprocess_docs import process_directory


def main() -> None:
    """
    流水线主函数
    执行顺序：
    1. 解析PDF为Markdown
    2. 预处理Markdown（清理无关内容）
    3. 用OneKE抽取三元组
    4. 用向量模型融合归一化
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="./data/raw")
    parser.add_argument("--parse-dir", default="./data/parsed")
    parser.add_argument("--preprocess-dir", default="./data/preprocessed")
    parser.add_argument("--raw-triples", default="./data/triples_raw/triples.jsonl")
    parser.add_argument("--fused-triples", default="./data/triples_fused/triples_fused.jsonl")
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析层")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理层")
    parser.add_argument("--keep-abstract", action="store_true", help="预处理时保留摘要")
    parser.add_argument("--schema-path", default="config/relation_types.json")
    parser.add_argument("--oneke-model", default="model/OneKE")
    parser.add_argument("--chunk-chars", type=int, default=150, help="文本块大小（极小chunk）")
    parser.add_argument("--overlap", type=int, default=30, help="重叠字符数（20%）")
    parser.add_argument("--embedding-model", default="model/Yuan-Embedding")
    parser.add_argument("--entity-threshold", type=float, default=0.85)
    parser.add_argument("--relation-threshold", type=float, default=0.9)
    args = parser.parse_args()

    # Step 1: 解析PDF为Markdown
    if not args.skip_parse:
        print("\n[Step 1/4] 解析PDF文档...")
        asyncio.run(run_parse(args.pdf_dir, args.parse_dir, skip_exists=True))
    
    # Step 2: 预处理Markdown（清理无关内容）
    if not args.skip_preprocess:
        print("\n[Step 2/4] 预处理文档...")
        process_directory(
            input_dir=args.parse_dir,
            output_dir=args.preprocess_dir,
            keep_abstract=args.keep_abstract,
        )
        extract_source_dir = args.preprocess_dir
    else:
        extract_source_dir = args.parse_dir
    
    # Step 3: 抽取三元组
    print("\n[Step 3/4] 抽取知识三元组...")
    raw_count = run_extract(
        parsed_dir=extract_source_dir,
        out_jsonl=args.raw_triples,
        schema_path=args.schema_path,
        model_path=args.oneke_model,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )
    
    # Step 4: 知识融合
    print("\n[Step 4/4] 知识融合与归一化...")
    fused_count = run_fusion(
        in_jsonl=args.raw_triples,
        out_jsonl=args.fused_triples,
        model_name=args.embedding_model,
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
        schema_path=args.schema_path,
    )
    
    print(f"\n[PIPELINE DONE] raw={raw_count}, fused={fused_count}")


if __name__ == "__main__":
    main()