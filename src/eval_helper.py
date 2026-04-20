"""
评估辅助工具
功能：
1. 提供评估工作流程的便捷入口
2. 检查评估数据的完整性
3. 生成评估报告摘要
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from common import load_jsonl


def check_annotation_status(csv_path: str) -> Dict:
    """
    检查CSV标注文件的完成状态
    Args:
        csv_path: CSV文件路径
    Returns:
        状态字典，包含总数、已标注数、未标注数
    """
    if not Path(csv_path).exists():
        return {
            "exists": False,
            "total": 0,
            "annotated": 0,
            "unannotated": 0,
            "completion_rate": 0.0
        }
    
    total = 0
    annotated = 0
    
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            is_correct = row.get("is_correct(1/0)", "").strip()
            if is_correct in ["0", "1", "0.0", "1.0"]:
                annotated += 1
    
    return {
        "exists": True,
        "total": total,
        "annotated": annotated,
        "unannotated": total - annotated,
        "completion_rate": annotated / total if total > 0 else 0.0
    }


def check_triples_file(triples_path: str) -> Dict:
    """
    检查三元组文件的基本信息
    Args:
        triples_path: 三元组JSONL文件路径
    Returns:
        信息字典
    """
    if not Path(triples_path).exists():
        return {
            "exists": False,
            "total_triples": 0,
            "total_entities": 0,
            "total_relations": 0
        }
    
    triples = load_jsonl(triples_path)
    
    # 统计实体和关系类型
    entities = set()
    relations = set()
    
    for t in triples:
        entities.add(t["head"])
        entities.add(t["tail"])
        relations.add(t["relation"])
    
    return {
        "exists": True,
        "total_triples": len(triples),
        "total_entities": len(entities),
        "total_relations": len(relations)
    }


def print_evaluation_status(
    entity_csv: str,
    relation_csv: str,
    triples_path: str
) -> None:
    """
    打印评估准备状态
    Args:
        entity_csv: 实体标注CSV路径
        relation_csv: 关系标注CSV路径
        triples_path: 三元组JSONL文件路径
    """
    print("=" * 80)
    print("评估准备状态检查")
    print("=" * 80)
    
    # 检查三元组文件
    print("\n【三元组数据】")
    triples_info = check_triples_file(triples_path)
    if triples_info["exists"]:
        print(f"✓ 文件存在: {triples_path}")
        print(f"  - 三元组总数: {triples_info['total_triples']}")
        print(f"  - 实体总数:   {triples_info['total_entities']}")
        print(f"  - 关系类型数: {triples_info['total_relations']}")
    else:
        print(f"✗ 文件不存在: {triples_path}")
    
    # 检查实体标注
    print("\n【实体标注】")
    entity_status = check_annotation_status(entity_csv)
    if entity_status["exists"]:
        print(f"✓ 文件存在: {entity_csv}")
        print(f"  - 总数:       {entity_status['total']}")
        print(f"  - 已标注:     {entity_status['annotated']}")
        print(f"  - 未标注:     {entity_status['unannotated']}")
        print(f"  - 完成率:     {entity_status['completion_rate']:.1%}")
        
        if entity_status['completion_rate'] < 1.0:
            print(f"  ⚠ 还有 {entity_status['unannotated']} 个实体未标注")
    else:
        print(f"✗ 文件不存在: {entity_csv}")
        print("  提示: 运行 python src/evaluate.py 生成标注模板")
    
    # 检查关系标注
    print("\n【关系标注】")
    relation_status = check_annotation_status(relation_csv)
    if relation_status["exists"]:
        print(f"✓ 文件存在: {relation_csv}")
        print(f"  - 总数:       {relation_status['total']}")
        print(f"  - 已标注:     {relation_status['annotated']}")
        print(f"  - 未标注:     {relation_status['unannotated']}")
        print(f"  - 完成率:     {relation_status['completion_rate']:.1%}")
        
        if relation_status['completion_rate'] < 1.0:
            print(f"  ⚠ 还有 {relation_status['unannotated']} 个关系未标注")
    else:
        print(f"✗ 文件不存在: {relation_csv}")
        print("  提示: 运行 python src/evaluate.py 生成标注模板")
    
    # 评估准备建议
    print("\n【评估准备建议】")
    print("-" * 80)
    
    if not triples_info["exists"]:
        print("1. 首先运行完整的数据处理流程生成三元组")
        print("   python src/pipeline.py")
    elif not entity_status["exists"] or not relation_status["exists"]:
        print("1. 生成标注模板（200个概念 + 400个关系）")
        print("   python src/evaluate.py --fused-triples data/triples_fused/triples_fused.jsonl")
    elif entity_status['completion_rate'] < 1.0 or relation_status['completion_rate'] < 1.0:
        print("1. 完成人工标注")
        print(f"   - 打开 {entity_csv}")
        print(f"   - 打开 {relation_csv}")
        print("   - 在 is_correct(1/0) 列填入 1（正确）或 0（错误）")
        print("   - 可选：在 comment 列填写错误原因")
    else:
        print("✓ 标注已完成，可以运行评估")
        print("\n2. 运行综合评估")
        print("   python src/eval_all.py \\")
        print(f"       --triples {triples_path} \\")
        print("       --schema config/relation_types.json \\")
        print(f"       --entity-csv {entity_csv} \\")
        print(f"       --relation-csv {relation_csv}")
    
    print("\n" + "=" * 80)


def generate_annotation_guide(output_path: str = "data/eval/annotation_guide.txt") -> None:
    """
    生成标注指南文档
    Args:
        output_path: 输出文件路径
    """
    from common import ensure_parent
    ensure_parent(output_path)
    
    guide_content = """
================================================================================
知识图谱标注指南
================================================================================

一、标注目的
-----------
对抽取的实体和关系进行人工标注，用于评估知识图谱的准确性。

二、标注规则
-----------

【实体标注】(sample_concepts.csv)
1. 检查实体是否是有效的概念
   - 正确示例：变体飞行器、气动特性、升阻比
   - 错误示例：并且、因此、所示（非实体）

2. 检查实体边界是否准确
   - 正确：变体飞行器
   - 错误：变体飞行器的（多余的"的"）

3. 在 is_correct(1/0) 列填写：
   - 1 或 1.0：实体正确
   - 0 或 0.0：实体错误

4. 在 comment 列填写错误原因（可选）：
   - 边界错误：实体边界不准确
   - 非实体：不是有效的概念
   - 其他：其他问题

【关系标注】(sample_relations.csv)
1. 检查三元组是否语义正确
   - 头实体、关系、尾实体三者是否构成有效的知识
   - 关系类型是否准确

2. 检查示例：
   正确：(变体飞行器, 具有, 气动特性)
   错误：(变体飞行器, 降低, 气动特性) - 关系类型错误

3. 在 is_correct(1/0) 列填写：
   - 1 或 1.0：关系正确
   - 0 或 0.0：关系错误

4. 在 comment 列填写错误原因（可选）：
   - 类型错误：关系类型不正确
   - 幻觉：原文中不存在的关系
   - 其他：其他问题

三、标注流程
-----------
1. 打开 CSV 文件（建议使用 Excel 或 WPS）
2. 逐行检查实体/关系
3. 在 is_correct(1/0) 列填入 1 或 0
4. 对于错误的项，在 comment 列简要说明原因
5. 保存文件（保持 UTF-8 编码）

四、注意事项
-----------
1. 标注时参考原始文档内容
2. 对于模糊的情况，可以标记为错误并在 comment 中说明
3. 保持标注的一致性
4. 定期保存文件，避免数据丢失

五、评估指标
-----------
标注完成后，系统将计算：
- 准确率 (Precision)：正确的比例
- 召回率 (Recall)：覆盖的比例（需要黄金标准）
- F1 分数：准确率和召回率的调和平均

================================================================================
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    print(f"标注指南已生成: {output_path}")


def main() -> None:
    """
    使用示例：
    # 检查评估准备状态
    python src/eval_helper.py --check
    
    # 生成标注指南
    python src/eval_helper.py --guide
    """
    parser = argparse.ArgumentParser(description="评估辅助工具")
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查评估准备状态"
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="生成标注指南文档"
    )
    parser.add_argument(
        "--entity-csv",
        default="data/eval/sample_concepts.csv",
        help="实体标注CSV路径"
    )
    parser.add_argument(
        "--relation-csv",
        default="data/eval/sample_relations.csv",
        help="关系标注CSV路径"
    )
    parser.add_argument(
        "--triples",
        default="data/triples_fused/triples_fused.jsonl",
        help="三元组JSONL文件路径"
    )
    
    args = parser.parse_args()
    
    if args.guide:
        generate_annotation_guide()
    
    if args.check:
        print_evaluation_status(
            entity_csv=args.entity_csv,
            relation_csv=args.relation_csv,
            triples_path=args.triples
        )
    
    if not args.check and not args.guide:
        # 默认行为：显示帮助信息
        parser.print_help()
        print("\n提示: 使用 --check 检查评估准备状态，或使用 --guide 生成标注指南")


if __name__ == "__main__":
    main()
