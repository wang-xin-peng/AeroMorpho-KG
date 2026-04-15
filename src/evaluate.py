import argparse
import csv
import random
from typing import Dict, List, Set, Tuple

from common import ensure_parent, load_jsonl


def sample_entities_relations(
    triples: List[Dict], concept_n: int = 200, relation_n: int = 400, seed: int = 42
) -> Tuple[List[str], List[Dict]]:
    random.seed(seed)
    entities: Set[str] = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    entity_list = sorted(list(entities))

    sampled_entities = (
        random.sample(entity_list, concept_n) if len(entity_list) >= concept_n else entity_list
    )
    sampled_relations = random.sample(triples, relation_n) if len(triples) >= relation_n else triples
    return sampled_entities, sampled_relations


def write_entities_csv(path: str, entities: List[str]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity", "is_correct(1/0)", "comment"])
        for e in entities:
            w.writerow([e, "", ""])


def write_relations_csv(path: str, triples: List[Dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head", "relation", "tail", "is_correct(1/0)", "comment"])
        for t in triples:
            w.writerow([t["head"], t["relation"], t["tail"], "", ""])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fused-triples", 
        required=False,  # 改为 False
        default="../data/triples_fused/triples_fused.jsonl",  # 添加默认值
        help="融合后的三元组 jsonl"
    )
    parser.add_argument(
        "--out-concepts", 
        required=False,
        default="../data/eval/sample_concepts.csv",
        help="概念抽样 CSV"
    )
    parser.add_argument(
        "--out-relations", 
        required=False,
        default="../data/eval/sample_relations.csv",
        help="关系抽样 CSV"
    )
    parser.add_argument("--concept-n", type=int, default=200)
    parser.add_argument("--relation-n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    triples = load_jsonl(args.fused_triples)
    if not triples:
        raise ValueError("融合三元组为空，无法抽样评估。")

    entities, rels = sample_entities_relations(
        triples, concept_n=args.concept_n, relation_n=args.relation_n, seed=args.seed
    )
    write_entities_csv(args.out_concepts, entities)
    write_relations_csv(args.out_relations, rels)
    print(
        f"[DONE] entity samples={len(entities)} -> {args.out_concepts}, "
        f"relation samples={len(rels)} -> {args.out_relations}"
    )


if __name__ == "__main__":
    main()
