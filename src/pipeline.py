"""
串联PDF解析、文档预处理、OneKE抽取、实体归一化和后处理的端到端流水线。
"""

import argparse
import asyncio

from extract_triples import run_extract
from normalize_and_filter import run_normalize_and_filter
from parse_docs import run_parse
from postprocess import run_postprocess
from preprocess_docs import process_directory


def main() -> None:
    """
    流水线主函数
    执行顺序：
    1. 解析PDF为Markdown
    2. 预处理Markdown（清理无关内容）
    3. 用OneKE抽取三元组
    4. 实体归一化和关系过滤
    5. 后处理（类型标注、约束检查、对称补全、互斥解决）
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="./data/raw")
    parser.add_argument("--parse-dir", default="./data/parsed")
    parser.add_argument("--preprocess-dir", default="./data/preprocessed")
    parser.add_argument("--raw-triples", default="./output/triples_raw/triples.jsonl")
    parser.add_argument("--normalized-triples", default="./output/triples_normalized/triples.jsonl")
    parser.add_argument("--final-triples", default="./output/triples_final/triples.jsonl")
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析层")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理层")
    parser.add_argument("--keep-abstract", action="store_true", help="预处理时保留摘要")
    parser.add_argument("--schema-path", default="config/relation_types.json")
    parser.add_argument("--entity-types-path", default="config/entity_types.json")
    parser.add_argument("--oneke-model", default="model/OneKE")
    parser.add_argument("--chunk-chars", type=int, default=150, help="文本块大小")
    parser.add_argument("--overlap", type=int, default=30, help="重叠字符数")
    parser.add_argument("--embedding-model", default="model/Yuan-Embedding")
    parser.add_argument("--entity-threshold", type=float, default=0.93)
    parser.add_argument("--type-threshold", type=float, default=0.6)
    args = parser.parse_args()

    # Step 1: 解析PDF为Markdown
    if not args.skip_parse:
        print("\n[Step 1/5] 解析PDF文档...")
        asyncio.run(run_parse(args.pdf_dir, args.parse_dir, skip_exists=True))
    
    # Step 2: 预处理Markdown（清理无关内容）
    if not args.skip_preprocess:
        print("\n[Step 2/5] 预处理文档...")
        run_preprocess(
            input_dir=args.parse_dir,
            output_dir=args.preprocess_dir,
            keep_abstract=args.keep_abstract,
        )
        extract_source_dir = args.preprocess_dir
    else:
        extract_source_dir = args.parse_dir
    
    # Step 3: 抽取三元组
    print("\n[Step 3/5] 抽取知识三元组...")
    raw_count = run_extract(
        parsed_dir=extract_source_dir,
        out_jsonl=args.raw_triples,
        schema_path=args.schema_path,
        model_path=args.oneke_model,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )
    
    # Step 4: 实体归一化和关系过滤
    print("\n[Step 4/5] 实体归一化和关系过滤...")
    normalized_count = run_normalize_and_filter(
        in_jsonl=args.raw_triples,
        out_jsonl=args.normalized_triples,
        model_name=args.embedding_model,
        entity_threshold=args.entity_threshold,
        schema_path=args.schema_path,
    )
    
    # Step 5: 后处理
    print("\n[Step 5/5] 后处理（类型标注、约束检查、对称补全、互斥解决）...")
    final_count = run_postprocess(
        in_jsonl=args.normalized_triples,
        out_jsonl=args.final_triples,
        entity_types_path=args.entity_types_path,
        relation_types_path=args.schema_path,
        embedding_model=args.embedding_model,
        type_threshold=args.type_threshold,
    )
    
    print(f"\n[PIPELINE DONE] raw={raw_count}, normalized={normalized_count}, final={final_count}")


if __name__ == "__main__":
    main()