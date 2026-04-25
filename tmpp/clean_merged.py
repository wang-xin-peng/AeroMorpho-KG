import json

input_path = r"d:\kg\AeroMorpho-KG\output\eval_result\merged.json"
output_path = r"d:\kg\AeroMorpho-KG\output\eval_result\merged_cleaned.json"

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entities = data.get('entities', [])
relations = data.get('relations', [])

entities_unique = list(set(entities))

entity_set = set(entities_unique)

filtered_relations = []
for rel in relations:
    if rel['head'] in entity_set and rel['tail'] in entity_set:
        filtered_relations.append(rel)

used_entities = set()
for rel in filtered_relations:
    used_entities.add(rel['head'])
    used_entities.add(rel['tail'])

final_entities = sorted([e for e in entities_unique if e in used_entities])

result = {
    'entities': final_entities,
    'relations': sorted(filtered_relations, key=lambda x: (x['head'], x['relation'], x['tail']))
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Entities: {len(entities)} -> dedup: {len(entities_unique)} -> filtered: {len(final_entities)}")
print(f"Relations: {len(relations)} -> filtered: {len(filtered_relations)}")
print(f"Saved to: {output_path}")