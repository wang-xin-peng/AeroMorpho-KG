import argparse
import asyncio

from extract_triples import run_extract
from fuse_entities import run_fusion
from parse_docs import run_parse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=False, default="../变构飞行器")  # 改成你的PDF目录
    parser.add_argument("--parse-dir", required=False, default="../data/parsed")  # 解析输出目录
    parser.add_argument("--raw-triples", required=False, default="../data/triples_raw/triples.jsonl")
    parser.add_argument("--fused-triples", required=False, default="../data/triples_fused/triples_fused.jsonl")
    parser.add_argument("--skip-parse", action="store_true", help="跳过解析层（已有 Markdown）")
    args = parser.parse_args()

    if not args.skip_parse:
        asyncio.run(run_parse(args.pdf_dir, args.parse_dir, skip_exists=True))
    raw_count = run_extract(args.parse_dir, args.raw_triples)
    fused_count = run_fusion(args.raw_triples, args.fused_triples)
    print(f"[PIPELINE DONE] raw={raw_count}, fused={fused_count}")


if __name__ == "__main__":
    main()