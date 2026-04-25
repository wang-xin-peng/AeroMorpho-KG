import json

ground_truth_path = r"d:\kg\AeroMorpho-KG\data\evaluation\ground_truth.json"
triples_path = r"d:\kg\AeroMorpho-KG\output\eval_result\triples_as_ground_truth.json"
output_path = r"d:\kg\AeroMorpho-KG\output\eval_result\diff_only_in_triples.json"

with open(ground_truth_path, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)

with open(triples_path, 'r', encoding='utf-8') as f:
    triples_data = json.load(f)

gt_entities = set(gt_data.get('entities', []))
gt_relations = set((r['head'], r['relation'], r['tail']) for r in gt_data.get('relations', []))

triples_entities = set(triples_data.get('entities', []))
triples_relations = set((r['head'], r['relation'], r['tail']) for r in triples_data.get('relations', []))

diff_entities = triples_entities - gt_entities
diff_relations = triples_relations - gt_relations

diff_relations_list = [{'head': h, 'relation': r, 'tail': t} for h, r, t in diff_relations]

result = {
    'entities': sorted(list(diff_entities)),
    'relations': sorted(diff_relations_list, key=lambda x: (x['head'], x['relation'], x['tail']))
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Entities only in triples: {len(diff_entities)}")
print(f"Relations only in triples: {len(diff_relations)}")
print(f"Result saved to: {output_path}")