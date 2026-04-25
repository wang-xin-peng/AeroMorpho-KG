"""
将triples_final.jsonl转换为ground_truth格式
"""

import json
from collections import Counter
from common import load_jsonl, ensure_parent


def convert_to_ground_truth(triples_path: str, output_path: str):
    triples = load_jsonl(triples_path)

    entities = []
    relations = []

    for t in triples:
        head = t["head"]
        tail = t["tail"]
        relation = t["relation"]

        if head not in entities:
            entities.append(head)
        if tail not in entities:
            entities.append(tail)

        relations.append({
            "head": head,
            "relation": relation,
            "tail": tail
        })

    result = {
        "entities": entities,
        "relations": relations
    }

    ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"转换完成：")
    print(f"  实体数量: {len(entities)}")
    print(f"  关系数量: {len(relations)}")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/eval_result/triples_final.jsonl")
    parser.add_argument("--output", default="output/eval_result/triples_as_ground_truth.json")
    args = parser.parse_args()

    convert_to_ground_truth(args.input, args.output)