import argparse
from typing import Dict, List

from rapidfuzz import fuzz

from common import dump_jsonl, load_jsonl


REL_NORMALIZE = {
    "具备": "具有",
    "拥有": "具有",
    "用于": "用于",
    "应用于": "用于",
    "可用于": "用于",
    "采用": "采用",
    "使用": "采用",
    "可分为": "可分为",
    "分为": "可分为",
    "包括": "包括",
    "包含": "包括",
}

ENTITY_ALIAS = {
    "变体飞行器": "变构飞行器",
    "空天变构飞行器": "变构飞行器",
    "跨域变构飞行器": "变构飞行器",
}


def normalize_relation(r: str) -> str:
    r = r.strip()
    return REL_NORMALIZE.get(r, r)


def pick_canonical(entity: str, canonical_list: List[str], threshold: int = 88) -> str:
    if entity in ENTITY_ALIAS:
        return ENTITY_ALIAS[entity]
    for c in canonical_list:
        if fuzz.ratio(entity, c) >= threshold:
            return c
    return entity


def run_fusion(in_jsonl: str, out_jsonl: str) -> int:
    triples = load_jsonl(in_jsonl)
    canonical_entities: List[str] = []
    fused: List[Dict] = []
    dedup = set()

    for t in triples:
        h0 = t["head"].strip()
        r0 = t["relation"].strip()
        ta0 = t["tail"].strip()

        h = pick_canonical(h0, canonical_entities)
        if h not in canonical_entities:
            canonical_entities.append(h)

        ta = pick_canonical(ta0, canonical_entities)
        if ta not in canonical_entities:
            canonical_entities.append(ta)

        r = normalize_relation(r0)
        key = (h, r, ta)
        if key in dedup:
            continue
        dedup.add(key)
        fused.append(
            {
                "head": h,
                "relation": r,
                "tail": ta,
                "source": t.get("source", ""),
                "method": t.get("method", ""),
            }
        )

    dump_jsonl(out_jsonl, fused)
    print(f"[DONE] fused triples = {len(fused)}, entities = {len(canonical_entities)}")
    print(f"[SAVE] {out_jsonl}")
    return len(fused)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-jsonl", required=True, help="原始三元组")
    parser.add_argument("--out-jsonl", required=True, help="融合后输出")
    args = parser.parse_args()
    run_fusion(args.in_jsonl, args.out_jsonl)


if __name__ == "__main__":
    main()
