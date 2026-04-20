"""
一致性评估工具
功能：
1. 检测实体一致性（同义词、消歧）
2. 检测关系一致性（对称性、传递性、互斥性）
3. 检测Schema一致性（类型匹配）
4. 生成一致性评估报告
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common import load_jsonl


def detect_entity_synonyms(triples: List[Dict], similarity_threshold: float = 0.8) -> List[Tuple[str, str]]:
    """
    检测可能的同义实体（基于字符串相似度）
    Args:
        triples: 三元组列表
        similarity_threshold: 相似度阈值
    Returns:
        可能的同义词对列表
    """
    # 收集所有实体
    entities = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    
    entity_list = sorted(list(entities))
    synonyms = []
    
    # 简单的字符串包含检测
    for i, e1 in enumerate(entity_list):
        for e2 in entity_list[i+1:]:
            # 检测包含关系
            if e1 in e2 or e2 in e1:
                if e1 != e2:
                    synonyms.append((e1, e2))
    
    return synonyms


def detect_symmetric_violations(triples: List[Dict], symmetric_relations: Set[str]) -> List[Dict]:
    """
    检测对称关系的违规情况
    Args:
        triples: 三元组列表
        symmetric_relations: 对称关系集合（如"对比"）
    Returns:
        违规的三元组列表
    """
    violations = []
    
    # 构建关系索引
    relation_index = defaultdict(set)
    for t in triples:
        if t["relation"] in symmetric_relations:
            relation_index[t["relation"]].add((t["head"], t["tail"]))
    
    # 检查对称性
    for relation, pairs in relation_index.items():
        for head, tail in pairs:
            # 检查反向关系是否存在
            if (tail, head) not in pairs:
                violations.append({
                    "type": "对称性违规",
                    "relation": relation,
                    "head": head,
                    "tail": tail,
                    "missing": f"{tail}-{relation}-{head}"
                })
    
    return violations


def detect_transitive_chains(triples: List[Dict], transitive_relations: Set[str]) -> List[Dict]:
    """
    检测传递关系链（用于发现可能缺失的关系）
    Args:
        triples: 三元组列表
        transitive_relations: 传递关系集合（如"包含"）
    Returns:
        可能缺失的传递关系列表
    """
    missing_relations = []
    
    for relation in transitive_relations:
        # 构建关系图
        graph = defaultdict(set)
        for t in triples:
            if t["relation"] == relation:
                graph[t["head"]].add(t["tail"])
        
        # 检查传递性：如果 A->B 且 B->C，则应该有 A->C
        for a in graph:
            for b in graph[a]:
                if b in graph:
                    for c in graph[b]:
                        # 检查 A->C 是否存在
                        if c not in graph[a]:
                            missing_relations.append({
                                "type": "传递性缺失",
                                "relation": relation,
                                "chain": f"{a}->{b}->{c}",
                                "missing": f"{a}-{relation}-{c}"
                            })
    
    return missing_relations


def detect_conflicting_relations(triples: List[Dict], conflict_pairs: List[Tuple[str, str]]) -> List[Dict]:
    """
    检测互斥关系（不应同时存在的关系）
    Args:
        triples: 三元组列表
        conflict_pairs: 互斥关系对列表（如[("优化", "降低")]）
    Returns:
        冲突的三元组对列表
    """
    conflicts = []
    
    # 构建实体对到关系的映射
    entity_pair_relations = defaultdict(set)
    for t in triples:
        key = (t["head"], t["tail"])
        entity_pair_relations[key].add(t["relation"])
    
    # 检查互斥关系
    for (head, tail), relations in entity_pair_relations.items():
        for r1, r2 in conflict_pairs:
            if r1 in relations and r2 in relations:
                conflicts.append({
                    "type": "关系互斥冲突",
                    "head": head,
                    "tail": tail,
                    "conflict_relations": [r1, r2]
                })
    
    return conflicts


def detect_schema_violations(triples: List[Dict], schema_path: str) -> List[Dict]:
    """
    检测Schema违规（关系类型是否在预定义的Schema中）
    Args:
        triples: 三元组列表
        schema_path: Schema配置文件路径
    Returns:
        违规的三元组列表
    """
    from schema import DeepKESchema
    
    violations = []
    
    # 加载Schema
    schema = DeepKESchema.from_json(schema_path)
    valid_relations = set(schema.get_relation_names())
    
    # 检查每个三元组的关系类型
    for t in triples:
        if t["relation"] not in valid_relations:
            violations.append({
                "type": "Schema违规",
                "head": t["head"],
                "relation": t["relation"],
                "tail": t["tail"],
                "reason": "关系类型不在预定义Schema中"
            })
    
    return violations


def run_consistency_eval(
    triples_path: str,
    schema_path: str,
    output_report: str = None
) -> Dict:
    """
    运行一致性评估
    Args:
        triples_path: 三元组JSONL文件路径
        schema_path: Schema配置文件路径
        output_report: 输出报告路径（可选）
    Returns:
        评估结果字典
    """
    # 加载三元组
    triples = load_jsonl(triples_path)
    
    results = {}
    
    # 1. 实体一致性检查
    synonyms = detect_entity_synonyms(triples)
    results["entity_synonyms"] = {
        "count": len(synonyms),
        "examples": synonyms[:10]  # 只保留前10个示例
    }
    
    # 2. 关系对称性检查
    symmetric_relations = {"对比"}  # 可配置
    symmetric_violations = detect_symmetric_violations(triples, symmetric_relations)
    results["symmetric_violations"] = {
        "count": len(symmetric_violations),
        "examples": symmetric_violations[:10]
    }
    
    # 3. 关系传递性检查
    transitive_relations = {"包含"}  # 可配置
    transitive_missing = detect_transitive_chains(triples, transitive_relations)
    results["transitive_missing"] = {
        "count": len(transitive_missing),
        "examples": transitive_missing[:10]
    }
    
    # 4. 关系互斥性检查
    conflict_pairs = [("优化", "降低"), ("提高", "降低")]  # 可配置
    conflicts = detect_conflicting_relations(triples, conflict_pairs)
    results["relation_conflicts"] = {
        "count": len(conflicts),
        "examples": conflicts[:10]
    }
    
    # 5. Schema一致性检查
    schema_violations = detect_schema_violations(triples, schema_path)
    results["schema_violations"] = {
        "count": len(schema_violations),
        "examples": schema_violations[:10]
    }
    
    # 生成报告
    if output_report:
        generate_consistency_report(results, output_report)
    
    return results


def generate_consistency_report(results: Dict, output_path: str) -> None:
    """
    生成一致性评估报告
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("一致性评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 实体一致性
        f.write("一、实体一致性\n")
        f.write("-" * 80 + "\n")
        synonyms = results["entity_synonyms"]
        f.write(f"可能的同义实体对数量: {synonyms['count']}\n\n")
        if synonyms['examples']:
            f.write("示例:\n")
            for e1, e2 in synonyms['examples'][:5]:
                f.write(f"  - '{e1}' ≈ '{e2}'\n")
        f.write("\n")
        
        # 关系对称性
        f.write("二、关系对称性\n")
        f.write("-" * 80 + "\n")
        symmetric = results["symmetric_violations"]
        f.write(f"对称性违规数量: {symmetric['count']}\n\n")
        if symmetric['examples']:
            f.write("示例:\n")
            for v in symmetric['examples'][:5]:
                f.write(f"  - 存在: {v['head']}-{v['relation']}-{v['tail']}\n")
                f.write(f"    缺失: {v['missing']}\n")
        f.write("\n")
        
        # 关系传递性
        f.write("三、关系传递性\n")
        f.write("-" * 80 + "\n")
        transitive = results["transitive_missing"]
        f.write(f"传递性缺失数量: {transitive['count']}\n\n")
        if transitive['examples']:
            f.write("示例:\n")
            for v in transitive['examples'][:5]:
                f.write(f"  - 链: {v['chain']}\n")
                f.write(f"    缺失: {v['missing']}\n")
        f.write("\n")
        
        # 关系互斥性
        f.write("四、关系互斥性\n")
        f.write("-" * 80 + "\n")
        conflicts = results["relation_conflicts"]
        f.write(f"互斥冲突数量: {conflicts['count']}\n\n")
        if conflicts['examples']:
            f.write("示例:\n")
            for v in conflicts['examples'][:5]:
                f.write(f"  - {v['head']} -> {v['tail']}\n")
                f.write(f"    冲突关系: {', '.join(v['conflict_relations'])}\n")
        f.write("\n")
        
        # Schema一致性
        f.write("五、Schema一致性\n")
        f.write("-" * 80 + "\n")
        schema = results["schema_violations"]
        f.write(f"Schema违规数量: {schema['count']}\n\n")
        if schema['examples']:
            f.write("示例:\n")
            for v in schema['examples'][:5]:
                f.write(f"  - {v['head']}-{v['relation']}-{v['tail']}\n")
                f.write(f"    原因: {v['reason']}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
        f.write("=" * 80 + "\n")


def main() -> None:
    """
    使用示例：
    python eval_consistency.py \
        --triples data/triples_fused/triples_fused.jsonl \
        --schema config/schema.json \
        --output-report data/eval/consistency_report.txt
    """
    parser = argparse.ArgumentParser(description="一致性评估工具")
    parser.add_argument("--triples", required=True, help="三元组JSONL文件路径")
    parser.add_argument("--schema", default="config/schema.json", help="Schema配置文件路径")
    parser.add_argument("--output-report", default="data/eval/consistency_report.txt", help="输出报告路径")
    args = parser.parse_args()
    
    results = run_consistency_eval(
        triples_path=args.triples,
        schema_path=args.schema,
        output_report=args.output_report
    )
    
    # 打印结果摘要
    print("\n" + "=" * 80)
    print("一致性评估结果摘要")
    print("=" * 80)
    print(f"\n可能的同义实体对:   {results['entity_synonyms']['count']}")
    print(f"对称性违规:         {results['symmetric_violations']['count']}")
    print(f"传递性缺失:         {results['transitive_missing']['count']}")
    print(f"关系互斥冲突:       {results['relation_conflicts']['count']}")
    print(f"Schema违规:         {results['schema_violations']['count']}")
    print(f"\n详细报告已保存至: {args.output_report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
