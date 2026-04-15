import argparse
from typing import Dict, List

from common import load_jsonl


def search_by_entity(triples: List[Dict], keyword: str, limit: int = 20) -> List[Dict]:
    rows = []
    for t in triples:
        if keyword in t["head"] or keyword in t["tail"]:
            rows.append(t)
        if len(rows) >= limit:
            break
    return rows


def relation_stat(triples: List[Dict], top_k: int = 20) -> List[tuple]:
    cnt = {}
    for t in triples:
        r = t["relation"]
        cnt[r] = cnt.get(r, 0) + 1
    return sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", required=True, help="融合后三元组 jsonl")
    parser.add_argument("--query", default="变构飞行器", help="查询关键词")
    args = parser.parse_args()

    triples = load_jsonl(args.triples)
    print(f"Loaded triples: {len(triples)}")
    print("\n[Top Relations]")
    for rel, n in relation_stat(triples):
        print(f"- {rel}: {n}")

    print(f"\n[Query: {args.query}]")
    rows = search_by_entity(triples, args.query, limit=20)
    if not rows:
        print("No matched triples.")
        return
    for i, t in enumerate(rows, start=1):
        print(f"{i}. ({t['head']}, {t['relation']}, {t['tail']})")


if __name__ == "__main__":
    main()
