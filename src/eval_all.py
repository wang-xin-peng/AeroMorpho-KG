"""
综合评估工具
功能：
1. 整合准确性、一致性、完整性三个维度的评估
2. 生成综合评估报告
3. 计算加权总分
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from eval_accuracy import run_accuracy_eval
from eval_consistency import run_consistency_eval
from eval_completeness import run_completeness_eval


def calculate_weighted_score(results: Dict, weights: Dict = None) -> float:
    """
    计算加权总分
    Args:
        results: 评估结果字典
        weights: 权重字典，默认为 {"accuracy": 0.5, "consistency": 0.2, "completeness": 0.3}
    Returns:
        加权总分（0-100）
    """
    if weights is None:
        weights = {
            "accuracy": 0.5,
            "consistency": 0.2,
            "completeness": 0.3
        }
    
    scores = {}
    
    # 准确性得分（基于F1或准确率）
    if "accuracy" in results and "relation" in results["accuracy"]:
        relation = results["accuracy"]["relation"]
        if relation["f1"] is not None:
            scores["accuracy"] = relation["f1"] * 100
        else:
            scores["accuracy"] = relation["precision"] * 100
    else:
        scores["accuracy"] = 0
    
    # 一致性得分（基于违规率的倒数）
    if "consistency" in results:
        consistency = results["consistency"]
        # 计算总违规数
        total_violations = (
            consistency["symmetric_violations"]["count"] +
            consistency["relation_conflicts"]["count"] +
            consistency["schema_violations"]["count"]
        )
        # 假设总三元组数
        total_triples = 1000  # 可以从结果中获取
        violation_rate = total_violations / total_triples if total_triples > 0 else 0
        scores["consistency"] = max(0, (1 - violation_rate) * 100)
    else:
        scores["consistency"] = 0
    
    # 完整性得分（基于覆盖率和连接度）
    if "completeness" in results:
        completeness = results["completeness"]
        relation_cov = completeness["relation_coverage"]["coverage_rate"]
        connectivity = completeness["entity_connectivity"]
        isolated_rate = connectivity["isolated_rate"]
        
        # 综合得分：关系覆盖率 * 0.6 + (1 - 孤立率) * 0.4
        scores["completeness"] = (relation_cov * 0.6 + (1 - isolated_rate) * 0.4) * 100
    else:
        scores["completeness"] = 0
    
    # 计算加权总分
    weighted_score = sum(scores[k] * weights[k] for k in weights.keys())
    
    return weighted_score, scores


def run_comprehensive_eval(
    triples_path: str,
    schema_path: str,
    entity_csv: str = None,
    relation_csv: str = None,
    output_dir: str = "data/eval"
) -> Dict:
    """
    运行综合评估
    Args:
        triples_path: 三元组JSONL文件路径
        schema_path: Schema配置文件路径
        entity_csv: 实体标注CSV路径（可选）
        relation_csv: 关系标注CSV路径（可选）
        output_dir: 输出目录
    Returns:
        综合评估结果字典
    """
    results = {}
    
    # 1. 准确性评估（如果提供了标注数据）
    if entity_csv and relation_csv:
        print("\n[1/3] 运行准确性评估...")
        accuracy_report = f"{output_dir}/accuracy_report.txt"
        results["accuracy"] = run_accuracy_eval(
            entity_csv=entity_csv,
            relation_csv=relation_csv,
            output_report=accuracy_report
        )
        print(f"✓ 准确性评估完成，报告保存至: {accuracy_report}")
    else:
        print("\n[1/3] 跳过准确性评估（未提供标注数据）")
    
    # 2. 一致性评估
    print("\n[2/3] 运行一致性评估...")
    consistency_report = f"{output_dir}/consistency_report.txt"
    results["consistency"] = run_consistency_eval(
        triples_path=triples_path,
        schema_path=schema_path,
        output_report=consistency_report
    )
    print(f"✓ 一致性评估完成，报告保存至: {consistency_report}")
    
    # 3. 完整性评估
    print("\n[3/3] 运行完整性评估...")
    completeness_report = f"{output_dir}/completeness_report.txt"
    results["completeness"] = run_completeness_eval(
        triples_path=triples_path,
        schema_path=schema_path,
        output_report=completeness_report
    )
    print(f"✓ 完整性评估完成，报告保存至: {completeness_report}")
    
    # 4. 生成综合报告
    print("\n[4/4] 生成综合评估报告...")
    comprehensive_report = f"{output_dir}/comprehensive_report.txt"
    generate_comprehensive_report(results, comprehensive_report)
    print(f"✓ 综合报告保存至: {comprehensive_report}")
    
    # 5. 保存JSON结果
    json_output = f"{output_dir}/evaluation_results.json"
    save_results_json(results, json_output)
    print(f"✓ JSON结果保存至: {json_output}")
    
    return results


def generate_comprehensive_report(results: Dict, output_path: str) -> None:
    """
    生成综合评估报告
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    # 计算加权总分
    weighted_score, scores = calculate_weighted_score(results)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("知识图谱综合评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体评分
        f.write("【总体评分】\n")
        f.write("-" * 80 + "\n")
        f.write(f"加权总分: {weighted_score:.2f} / 100\n\n")
        
        f.write("各维度得分:\n")
        for dimension, score in scores.items():
            f.write(f"  - {dimension.capitalize()}: {score:.2f}\n")
        f.write("\n")
        
        # 准确性摘要
        if "accuracy" in results:
            f.write("【准确性评估摘要】\n")
            f.write("-" * 80 + "\n")
            
            if "entity" in results["accuracy"]:
                entity = results["accuracy"]["entity"]
                f.write(f"实体准确率: {entity['precision']:.2%}\n")
                if entity['f1'] is not None:
                    f.write(f"实体 F1:     {entity['f1']:.2%}\n")
            
            if "relation" in results["accuracy"]:
                relation = results["accuracy"]["relation"]
                f.write(f"关系准确率: {relation['precision']:.2%}\n")
                if relation['f1'] is not None:
                    f.write(f"关系 F1:     {relation['f1']:.2%}\n")
            f.write("\n")
        
        # 一致性摘要
        if "consistency" in results:
            f.write("【一致性评估摘要】\n")
            f.write("-" * 80 + "\n")
            consistency = results["consistency"]
            f.write(f"可能的同义实体对:   {consistency['entity_synonyms']['count']}\n")
            f.write(f"对称性违规:         {consistency['symmetric_violations']['count']}\n")
            f.write(f"关系互斥冲突:       {consistency['relation_conflicts']['count']}\n")
            f.write(f"Schema违规:         {consistency['schema_violations']['count']}\n")
            f.write("\n")
        
        # 完整性摘要
        if "completeness" in results:
            f.write("【完整性评估摘要】\n")
            f.write("-" * 80 + "\n")
            completeness = results["completeness"]
            entity_cov = completeness["entity_coverage"]
            relation_cov = completeness["relation_coverage"]
            connectivity = completeness["entity_connectivity"]
            density = completeness["knowledge_density"]
            
            f.write(f"实体总数:           {entity_cov['total_entities']}\n")
            f.write(f"关系总数:           {density['relation_count']}\n")
            f.write(f"关系覆盖率:         {relation_cov['coverage_rate']:.2%}\n")
            f.write(f"孤立实体比例:       {connectivity['isolated_rate']:.2%}\n")
            f.write(f"平均连接度:         {connectivity['avg_connectivity']:.2f}\n")
            f.write(f"知识图谱密度:       {density['density']:.6f}\n")
            f.write("\n")
        
        # 改进建议
        f.write("【改进建议】\n")
        f.write("-" * 80 + "\n")
        suggestions = generate_suggestions(results, scores)
        for i, suggestion in enumerate(suggestions, 1):
            f.write(f"{i}. {suggestion}\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
        f.write("=" * 80 + "\n")


def generate_suggestions(results: Dict, scores: Dict) -> list:
    """
    根据评估结果生成改进建议
    Args:
        results: 评估结果字典
        scores: 各维度得分
    Returns:
        建议列表
    """
    suggestions = []
    
    # 准确性建议
    if "accuracy" in scores and scores["accuracy"] < 70:
        suggestions.append("准确率较低，建议：(1) 优化Schema定义，增加description；(2) 调整模型参数；(3) 增加训练数据")
    
    # 一致性建议
    if "consistency" in results:
        consistency = results["consistency"]
        if consistency["entity_synonyms"]["count"] > 50:
            suggestions.append("检测到大量可能的同义实体，建议进行实体融合和消歧处理")
        if consistency["schema_violations"]["count"] > 0:
            suggestions.append(f"存在{consistency['schema_violations']['count']}个Schema违规，建议检查抽取逻辑")
    
    # 完整性建议
    if "completeness" in results:
        completeness = results["completeness"]
        relation_cov = completeness["relation_coverage"]
        connectivity = completeness["entity_connectivity"]
        
        if relation_cov["coverage_rate"] < 0.5:
            suggestions.append(f"关系覆盖率仅{relation_cov['coverage_rate']:.1%}，建议增加文本数据或优化抽取策略")
        
        if connectivity["isolated_rate"] > 0.3:
            suggestions.append(f"孤立实体比例达{connectivity['isolated_rate']:.1%}，建议检查实体抽取的完整性")
    
    if not suggestions:
        suggestions.append("整体表现良好，继续保持！")
    
    return suggestions


def save_results_json(results: Dict, output_path: str) -> None:
    """
    保存评估结果为JSON格式
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    """
    使用示例：
    python eval_all.py \
        --triples data/triples_fused/triples_fused.jsonl \
        --schema config/relation_types.json \
        --entity-csv data/eval/sample_concepts.csv \
        --relation-csv data/eval/sample_relations.csv \
        --output-dir data/eval
    """
    parser = argparse.ArgumentParser(description="综合评估工具")
    parser.add_argument("--triples", required=True, help="三元组JSONL文件路径")
    parser.add_argument("--schema", default="config/relation_types.json", help="关系类型配置文件路径")
    parser.add_argument("--entity-csv", help="实体标注CSV路径（可选）")
    parser.add_argument("--relation-csv", help="关系标注CSV路径（可选）")
    parser.add_argument("--output-dir", default="data/eval", help="输出目录")
    args = parser.parse_args()
    
    print("=" * 80)
    print("开始综合评估")
    print("=" * 80)
    
    results = run_comprehensive_eval(
        triples_path=args.triples,
        schema_path=args.schema,
        entity_csv=args.entity_csv,
        relation_csv=args.relation_csv,
        output_dir=args.output_dir
    )
    
    # 计算并显示总分
    weighted_score, scores = calculate_weighted_score(results)
    
    print("\n" + "=" * 80)
    print("综合评估完成")
    print("=" * 80)
    print(f"\n【总分】 {weighted_score:.2f} / 100\n")
    print("【各维度得分】")
    for dimension, score in scores.items():
        print(f"  - {dimension.capitalize()}: {score:.2f}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
