"""
准确性评估工具
功能：
1. 读取人工标注的CSV文件
2. 计算实体和关系的准确率、召回率、F1分数
3. 生成准确性评估报告
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from common import load_jsonl


def load_annotated_csv(csv_path: str, is_entity: bool = False) -> Tuple[List[Dict], int, int]:
    """
    加载人工标注的CSV文件
    Args:
        csv_path: CSV文件路径
        is_entity: 是否为实体CSV（True）还是关系CSV（False）
    Returns:
        (标注数据列表, 正确数量, 总数量)
    """
    data = []
    correct_count = 0
    total_count = 0
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            is_correct_str = row.get("is_correct(1/0)", "").strip()
            
            # 解析标注结果
            is_correct = None
            if is_correct_str in ["1", "1.0"]:
                is_correct = True
                correct_count += 1
            elif is_correct_str in ["0", "0.0"]:
                is_correct = False
            
            if is_entity:
                data.append({
                    "entity": row.get("entity", "").strip(),
                    "is_correct": is_correct,
                    "comment": row.get("comment", "").strip()
                })
            else:
                data.append({
                    "head": row.get("head", "").strip(),
                    "relation": row.get("relation", "").strip(),
                    "tail": row.get("tail", "").strip(),
                    "is_correct": is_correct,
                    "comment": row.get("comment", "").strip()
                })
    
    return data, correct_count, total_count


def calculate_precision(correct: int, total: int) -> float:
    """
    计算准确率（Precision）
    Args:
        correct: 正确数量
        total: 总数量
    Returns:
        准确率（0-1之间的浮点数）
    """
    return correct / total if total > 0 else 0.0


def calculate_recall(extracted_correct: int, gold_total: int) -> float:
    """
    计算召回率（Recall）
    Args:
        extracted_correct: 抽取正确的数量
        gold_total: 黄金标准中的总数量
    Returns:
        召回率（0-1之间的浮点数）
    """
    return extracted_correct / gold_total if gold_total > 0 else 0.0


def calculate_f1(precision: float, recall: float) -> float:
    """
    计算F1分数
    Args:
        precision: 准确率
        recall: 召回率
    Returns:
        F1分数（0-1之间的浮点数）
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def analyze_error_types(data: List[Dict], is_entity: bool = False) -> Dict[str, int]:
    """
    分析错误类型分布
    Args:
        data: 标注数据列表
        is_entity: 是否为实体数据
    Returns:
        错误类型统计字典
    """
    error_types = {
        "边界错误": 0,      # 实体边界不准确
        "类型错误": 0,      # 关系类型错误
        "幻觉": 0,          # 不存在的实体/关系
        "其他": 0
    }
    
    for item in data:
        if item["is_correct"] is False:
            comment = item.get("comment", "").lower()
            if "边界" in comment or "boundary" in comment:
                error_types["边界错误"] += 1
            elif "类型" in comment or "type" in comment or "关系" in comment:
                error_types["类型错误"] += 1
            elif "幻觉" in comment or "hallucination" in comment or "不存在" in comment:
                error_types["幻觉"] += 1
            else:
                error_types["其他"] += 1
    
    return error_types


def run_accuracy_eval(
    entity_csv: str,
    relation_csv: str,
    gold_entity_count: int = None,
    gold_relation_count: int = None,
    output_report: str = None
) -> Dict:
    """
    运行准确性评估
    Args:
        entity_csv: 实体标注CSV路径
        relation_csv: 关系标注CSV路径
        gold_entity_count: 黄金标准中的实体总数（用于计算召回率）
        gold_relation_count: 黄金标准中的关系总数（用于计算召回率）
        output_report: 输出报告路径（可选）
    Returns:
        评估结果字典
    """
    results = {}
    
    # 评估实体准确性
    if Path(entity_csv).exists():
        entity_data, entity_correct, entity_total = load_annotated_csv(entity_csv, is_entity=True)
        entity_precision = calculate_precision(entity_correct, entity_total)
        
        # 如果提供了黄金标准数量，计算召回率和F1
        entity_recall = None
        entity_f1 = None
        if gold_entity_count is not None:
            entity_recall = calculate_recall(entity_correct, gold_entity_count)
            entity_f1 = calculate_f1(entity_precision, entity_recall)
        
        entity_errors = analyze_error_types(entity_data, is_entity=True)
        
        results["entity"] = {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1,
            "correct": entity_correct,
            "total": entity_total,
            "error_types": entity_errors
        }
    
    # 评估关系准确性
    if Path(relation_csv).exists():
        relation_data, relation_correct, relation_total = load_annotated_csv(relation_csv, is_entity=False)
        relation_precision = calculate_precision(relation_correct, relation_total)
        
        # 如果提供了黄金标准数量，计算召回率和F1
        relation_recall = None
        relation_f1 = None
        if gold_relation_count is not None:
            relation_recall = calculate_recall(relation_correct, gold_relation_count)
            relation_f1 = calculate_f1(relation_precision, relation_recall)
        
        relation_errors = analyze_error_types(relation_data, is_entity=False)
        
        results["relation"] = {
            "precision": relation_precision,
            "recall": relation_recall,
            "f1": relation_f1,
            "correct": relation_correct,
            "total": relation_total,
            "error_types": relation_errors
        }
    
    # 生成报告
    if output_report:
        generate_accuracy_report(results, output_report)
    
    return results


def generate_accuracy_report(results: Dict, output_path: str) -> None:
    """
    生成准确性评估报告
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("准确性评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 实体评估结果
        if "entity" in results:
            entity = results["entity"]
            f.write("一、实体准确性评估\n")
            f.write("-" * 80 + "\n")
            f.write(f"准确率 (Precision): {entity['precision']:.2%}\n")
            if entity['recall'] is not None:
                f.write(f"召回率 (Recall):    {entity['recall']:.2%}\n")
                f.write(f"F1 分数:            {entity['f1']:.2%}\n")
            f.write(f"正确数量:           {entity['correct']}/{entity['total']}\n\n")
            
            f.write("错误类型分布:\n")
            for error_type, count in entity['error_types'].items():
                if count > 0:
                    f.write(f"  - {error_type}: {count}\n")
            f.write("\n")
        
        # 关系评估结果
        if "relation" in results:
            relation = results["relation"]
            f.write("二、关系准确性评估\n")
            f.write("-" * 80 + "\n")
            f.write(f"准确率 (Precision): {relation['precision']:.2%}\n")
            if relation['recall'] is not None:
                f.write(f"召回率 (Recall):    {relation['recall']:.2%}\n")
                f.write(f"F1 分数:            {relation['f1']:.2%}\n")
            f.write(f"正确数量:           {relation['correct']}/{relation['total']}\n\n")
            
            f.write("错误类型分布:\n")
            for error_type, count in relation['error_types'].items():
                if count > 0:
                    f.write(f"  - {error_type}: {count}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
        f.write("=" * 80 + "\n")


def main() -> None:
    """
    使用示例：
    python eval_accuracy.py \
        --entity-csv data/eval/sample_concepts.csv \
        --relation-csv data/eval/sample_relations.csv \
        --output-report data/eval/accuracy_report.txt
    """
    parser = argparse.ArgumentParser(description="准确性评估工具")
    parser.add_argument("--entity-csv", required=True, help="实体标注CSV路径")
    parser.add_argument("--relation-csv", required=True, help="关系标注CSV路径")
    parser.add_argument("--gold-entity-count", type=int, help="黄金标准中的实体总数")
    parser.add_argument("--gold-relation-count", type=int, help="黄金标准中的关系总数")
    parser.add_argument("--output-report", default="data/eval/accuracy_report.txt", help="输出报告路径")
    args = parser.parse_args()
    
    results = run_accuracy_eval(
        entity_csv=args.entity_csv,
        relation_csv=args.relation_csv,
        gold_entity_count=args.gold_entity_count,
        gold_relation_count=args.gold_relation_count,
        output_report=args.output_report
    )
    
    # 打印结果摘要
    print("\n" + "=" * 80)
    print("准确性评估结果摘要")
    print("=" * 80)
    
    if "entity" in results:
        entity = results["entity"]
        print(f"\n实体准确率: {entity['precision']:.2%} ({entity['correct']}/{entity['total']})")
        if entity['f1'] is not None:
            print(f"实体 F1:     {entity['f1']:.2%}")
    
    if "relation" in results:
        relation = results["relation"]
        print(f"\n关系准确率: {relation['precision']:.2%} ({relation['correct']}/{relation['total']})")
        if relation['f1'] is not None:
            print(f"关系 F1:     {relation['f1']:.2%}")
    
    print(f"\n详细报告已保存至: {args.output_report}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
