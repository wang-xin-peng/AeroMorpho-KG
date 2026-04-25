import json

file1 = r"d:\kg\AeroMorpho-KG\output\eval_result\diff_only_in_triples_cleaned.json"
file2 = r"d:\kg\AeroMorpho-KG\data\evaluation\ground_truth.json"
output_path = r"d:\kg\AeroMorpho-KG\output\eval_result\merged.json"

with open(file1, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

with open(file2, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

entities1 = set(data1.get('entities', []))
entities2 = set(data2.get('entities', []))

merged_entities = entities1 | entities2

relations1 = {(r['head'], r['relation'], r['tail']) for r in data1.get('relations', [])}
relations2 = {(r['head'], r['relation'], r['tail']) for r in data2.get('relations', [])}

merged_relations = relations1 | relations2

result = {
    'entities': sorted(list(merged_entities)),
    'relations': sorted([{'head': h, 'relation': r, 'tail': t} for h, r, t in merged_relations],
                        key=lambda x: (x['head'], x['relation'], x['tail']))
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Entities: {len(entities1)} + {len(entities2)} = {len(merged_entities)}")
print(f"Relations: {len(relations1)} + {len(relations2)} = {len(merged_relations)}")
print(f"Merged result saved to: {output_path}")