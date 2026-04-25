import json

input_path = r"d:\kg\AeroMorpho-KG\output\eval_result\diff_only_in_triples.json"
output_path = r"d:\kg\AeroMorpho-KG\output\eval_result\diff_only_in_triples_cleaned.json"

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
    'relations': filtered_relations
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Original entities: {len(entities)} -> After dedup: {len(entities_unique)} -> After filter: {len(final_entities)}")
print(f"Original relations: {len(relations)} -> After filter: {len(filtered_relations)}")
print(f"Result saved to: {output_path}")