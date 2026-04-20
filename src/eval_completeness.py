"""
完整性评估工具
功能：
1. 评估实体覆盖度（各类型实体数量分布）
2. 评估关系覆盖度（各类型关系数量分布）
3. 评估知识密度（实体连接度、孤立实体比例）
4. 生成完整性评估报告和可视化图表
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common import load_jsonl


def calculate_entity_coverage(triples: List[Dict]) -> Dict:
    """
    计算实体覆盖度统计
    Args:
        triples: 三元组列表
    Returns:
        实体覆盖度统计字典
    """
    # 收集所有实体
    entities = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    
    # 统计每个实体出现的次数
    entity_freq = Counter()
    for t in triples:
        entity_freq[t["head"]] += 1
        entity_freq[t["tail"]] += 1
    
    # 统计来源文档覆盖
    source_coverage = Counter()
    for t in triples:
        if "source" in t:
            source_coverage[t["source"]] += 1
    
    return {
        "total_entities": len(entities),
        "total_mentions": sum(entity_freq.values()),
        "avg_mentions_per_entity": sum(entity_freq.values()) / len(entities) if entities else 0,
        "source_coverage": dict(source_coverage),
        "top_entities": entity_freq.most_common(10)
    }


def calculate_relation_coverage(triples: List[Dict], schema_path: str) -> Dict:
    """
    计算关系覆盖度统计
    Args:
        triples: 三元组列表
        schema_path: Schema配置文件路径
    Returns:
        关系覆盖度统计字典
    """
    from schema import DeepKESchema
    
    # 加载Schema
    schema = DeepKESchema.from_json(schema_path)
    all_relations = set(schema.get_relation_names())
    
    # 统计每种关系的数量
    relation_freq = Counter()
    for t in triples:
        relation_freq[t["relation"]] += 1
    
    # 计算覆盖率
    used_relations = set(relation_freq.keys())
    coverage_rate = len(used_relations) / len(all_relations) if all_relations else 0
    
    # 未使用的关系
    unused_relations = all_relations - used_relations
    
    return {
        "total_relation_types": len(all_relations),
        "used_relation_types": len(used_relations),
        "coverage_rate": coverage_rate,
        "unused_relations": sorted(list(unused_relations)),
        "relation_distribution": dict(relation_freq),
        "top_relations": relation_freq.most_common(10)
    }


def calculate_entity_connectivity(triples: List[Dict]) -> Dict:
    """
    计算实体连接度统计
    Args:
        triples: 三元组列表
    Returns:
        连接度统计字典
    """
    # 构建实体连接图
    entity_connections = defaultdict(set)
    all_entities = set()
    
    for t in triples:
        head = t["head"]
        tail = t["tail"]
        all_entities.add(head)
        all_entities.add(tail)
        entity_connections[head].add(tail)
        entity_connections[tail].add(head)
    
    # 计算连接度分布
    connectivity = [len(entity_connections[e]) for e in all_entities]
    
    # 孤立实体（连接度为0）
    isolated_entities = [e for e in all_entities if len(entity_connections[e]) == 0]
    
    # 连接度统计
    if connectivity:
        avg_connectivity = sum(connectivity) / len(connectivity)
        max_connectivity = max(connectivity)
        min_connectivity = min(connectivity)
    else:
        avg_connectivity = max_connectivity = min_connectivity = 0
    
    # 连接度分布区间
    connectivity_bins = {
        "0": 0,
        "1-5": 0,
        "6-10": 0,
        "11-20": 0,
        "20+": 0
    }
    
    for conn in connectivity:
        if conn == 0:
            connectivity_bins["0"] += 1
        elif conn <= 5:
            connectivity_bins["1-5"] += 1
        elif conn <= 10:
            connectivity_bins["6-10"] += 1
        elif conn <= 20:
            connectivity_bins["11-20"] += 1
        else:
            connectivity_bins["20+"] += 1
    
    return {
        "total_entities": len(all_entities),
        "isolated_entities": len(isolated_entities),
        "isolated_rate": len(isolated_entities) / len(all_entities) if all_entities else 0,
        "avg_connectivity": avg_connectivity,
        "max_connectivity": max_connectivity,
        "min_connectivity": min_connectivity,
        "connectivity_distribution": connectivity_bins
    }


def calculate_knowledge_density(triples: List[Dict]) -> Dict:
    """
    计算知识密度指标
    Args:
        triples: 三元组列表
    Returns:
        知识密度统计字典
    """
    # 收集所有实体
    entities = set()
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
    
    entity_count = len(entities)
    relation_count = len(triples)
    
    # 理论最大关系数（完全图）
    max_possible_relations = entity_count * (entity_count - 1) if entity_count > 1 else 0
    
    # 实际密度
    density = relation_count / max_possible_relations if max_possible_relations > 0 else 0
    
    # 实体-关系比例
    entity_relation_ratio = entity_count / relation_count if relation_count > 0 else 0
    
    return {
        "entity_count": entity_count,
        "relation_count": relation_count,
        "max_possible_relations": max_possible_relations,
        "density": density,
        "entity_relation_ratio": entity_relation_ratio
    }


def run_completeness_eval(
    triples_path: str,
    schema_path: str,
    output_report: str = None
) -> Dict:
    """
    运行完整性评估
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
    
    # 1. 实体覆盖度
    results["entity_coverage"] = calculate_entity_coverage(triples)
    
    # 2. 关系覆盖度
    results["relation_coverage"] = calculate_relation_coverage(triples, schema_path)
    
    # 3. 实体连接度
    results["entity_connectivity"] = calculate_entity_connectivity(triples)
    
    # 4. 知识密度
    results["knowledge_density"] = calculate_knowledge_density(triples)
    
    # 生成报告
    if output_report:
        generate_completeness_report(results, output_report)
    
    return results


def generate_completeness_report(results: Dict, output_path: str) -> None:
    """
    生成完整性评估报告
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("完整性评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 实体覆盖度
        f.write("一、实体覆盖度\n")
        f.write("-" * 80 + "\n")
        entity_cov = results["entity_coverage"]
        f.write(f"实体总数:           {entity_cov['total_entities']}\n")
        f.write(f"实体提及总数:       {entity_cov['total_mentions']}\n")
        f.write(f"平均每实体提及次数: {entity_cov['avg_mentions_per_entity']:.2f}\n\n")
        
        f.write("文档覆盖度:\n")
        for source, count in sorted(entity_cov['source_coverage'].items(), key=lambda x: x[1], reverse=True)[:5]:
            f.write(f"  - {source}: {count} 个三元组\n")
        
        f.write("\n高频实体 Top 10:\n")
        for entity, freq in entity_cov['top_entities']:
            f.write(f"  - {entity}: {freq} 次\n")
        f.write("\n")
        
        # 关系覆盖度
        f.write("二、关系覆盖度\n")
        f.write("-" * 80 + "\n")
        relation_cov = results["relation_coverage"]
        f.write(f"Schema定义关系数:   {relation_cov['total_relation_types']}\n")
        f.write(f"实际使用关系数:     {relation_cov['used_relation_types']}\n")
        f.write(f"关系覆盖率:         {relation_cov['coverage_rate']:.2%}\n\n")
        
        if relation_cov['unused_relations']:
            f.write(f"未使用的关系 ({len(relation_cov['unused_relations'])}个):\n")
            for rel in relation_cov['unused_relations'][:10]:
                f.write(f"  - {rel}\n")
            if len(relation_cov['unused_relations']) > 10:
                f.write(f"  ... 还有 {len(relation_cov['unused_relations']) - 10} 个\n")
        
        f.write("\n关系分布 Top 10:\n")
        for relation, count in relation_cov['top_relations']:
            f.write(f"  - {relation}: {count} 个三元组\n")
        f.write("\n")
        
        # 实体连接度
        f.write("三、实体连接度\n")
        f.write("-" * 80 + "\n")
        connectivity = results["entity_connectivity"]
        f.write(f"实体总数:           {connectivity['total_entities']}\n")
        f.write(f"孤立实体数:         {connectivity['isolated_entities']}\n")
        f.write(f"孤立实体比例:       {connectivity['isolated_rate']:.2%}\n")
        f.write(f"平均连接度:         {connectivity['avg_connectivity']:.2f}\n")
        f.write(f"最大连接度:         {connectivity['max_connectivity']}\n")
        f.write(f"最小连接度:         {connectivity['min_connectivity']}\n\n")
        
        f.write("连接度分布:\n")
        for bin_range, count in connectivity['connectivity_distribution'].items():
            f.write(f"  - {bin_range}: {count} 个实体\n")
        f.write("\n")
        
        # 知识密度
        f.write("四、知识密度\n")
        f.write("-" * 80 + "\n")
        density = results["knowledge_density"]
        f.write(f"实体数量:           {density['entity_count']}\n")
        f.write(f"关系数量:           {density['relation_count']}\n")
        f.write(f"理论最大关系数:     {density['max_possible_relations']}\n")
        f.write(f"知识图谱密度:       {density['density']:.6f}\n")
        f.write(f"实体-关系比:        {density['entity_relation_ratio']:.2f}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
        f.write("=" * 80 + "\n")


def main() -> None:
    """
    使用示例：
    python eval_completeness.py \
        --triples data/triples_fused/triples_fused.jsonl \
        --schema config/schema.json \
        --output-report data/eval/completeness_report.txt
    """
    parser = argparse.ArgumentParser(description="完整性评估工具")
    parser.add_argument("--triples", required=True, help="三元组JSONL文件路径")
    parser.add_argument("--schema", default="config/schema.json", help="Schema配置文件路径")
    parser.add_argument("--output-report", default="data/eval/completeness_report.txt", help="输出报告路径")
    args = parser.parse_args()
    
    results = run_completeness_eval(
        triples_path=args.triples,
        schema_path=args.schema,
        output_report=args.output_report
    )
    
    # 打印结果摘要
    print("\n" + "=" * 80)
    print("完整性评估结果摘要")
    print("=" * 80)
    
    entity_cov = results["entity_coverage"]
    relation_cov = results["relation_coverage"]
    connectivity = results["entity_connectivity"]
    density = results["knowledge_density"]
    
    print(f"\n实体总数:           {entity_cov['total_entities']}")
    print(f"关系总数:           {density['relation_count']}")
    print(f"关系覆盖率:         {relation_cov['coverage_rate']:.2%}")
    print(f"孤立实体比例:       {connectivity['isolated_rate']:.2%}")
    print(f"平均连接度:         {connectivity['avg_connectivity']:.2f}")
    print(f"知识图谱密度:       {density['density']:.6f}")
    
    print(f"\n详细报告已保存至: {args.output_report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
