# 串联解析、OneKE抽取和BGE融合的端到端流水线
import argparse
import asyncio

from extract_triples_oneke import run_extract_oneke
from fuse_entities_bge import run_fusion_bge
from parse_docs import run_parse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--parse-dir", required=True)
    parser.add_argument("--raw-triples", required=True)
    parser.add_argument("--fused-triples", required=True)
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析层（已有 Markdown）")
    parser.add_argument("--schema-path", default="config/schema.json")
    parser.add_argument("--oneke-model", default="zjunlp/OneKE")
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--bge-model", default="IEITYuan/Yuan-embedding-2.0-zh")
    parser.add_argument("--entity-threshold", type=float, default=0.85)
    parser.add_argument("--relation-threshold", type=float, default=0.9)
    args = parser.parse_args()

    if not args.skip_parse:
        asyncio.run(run_parse(args.pdf_dir, args.parse_dir, skip_exists=True))
    raw_count = run_extract_oneke(
        parsed_dir=args.parse_dir,
        out_jsonl=args.raw_triples,
        schema_path=args.schema_path,
        model_path=args.oneke_model,
        chunk_chars=args.chunk_chars,
    )
    fused_count = run_fusion_bge(
        in_jsonl=args.raw_triples,
        out_jsonl=args.fused_triples,
        model_name=args.bge_model,
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
    )
    print(f"[PIPELINE DONE] raw={raw_count}, fused={fused_count}")


if __name__ == "__main__":
    main()