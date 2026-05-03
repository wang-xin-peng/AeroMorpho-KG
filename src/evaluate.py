"""
综合评估工具
功能：
1. 在eval.md上进行提取
2. 与ground_truth.json对比进行三方面评估:
    准确性: 实体和关系的精确率、召回率、F1
    一致性: Schema一致性、对称性、互斥性
    完整性：实体覆盖率、关系覆盖率、知识密度
3. 输出评估结果到output/eval_result/...
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
from common import ensure_parent, load_jsonl, dump_jsonl
from extract_triples import run_extract
from normalize_and_filter import run_normalize_and_filter
from postprocess import run_postprocess


def load_ground_truth(gt_path: str) -> Dict:
    """加载人工标注的答案"""
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_entity(name: str) -> str:
    """实体名称归一化（去除空格、转小写用于匹配）"""
    return name.strip().lower()


def evaluate_accuracy(
    predicted_triples: List[Dict],
    ground_truth: Dict
) -> Dict:
    """
    准确性评估：计算实体和关系的精确率、召回率、F1
    """
    # 构建人工标注的答案集合 - entities是字符串数组
    entities_data = ground_truth["entities"]
    gt_entities = {normalize_entity(e) for e in entities_data}
    gt_relations = set()
    for r in ground_truth["relations"]:
        key = (normalize_entity(r["head"]), r["relation"], normalize_entity(r["tail"]))
        gt_relations.add(key)
    
    # 预测实体（从三元组中提取）
    pred_entities: Set[str] = set()
    for t in predicted_triples:
        pred_entities.add(normalize_entity(t["head"]))
        pred_entities.add(normalize_entity(t["tail"]))
    
    # 预测关系
    pred_relations = set()
    for t in predicted_triples:
        key = (normalize_entity(t["head"]), t["relation"], normalize_entity(t["tail"]))
        pred_relations.add(key)
    
    # 实体评估
    entity_correct = len(pred_entities & gt_entities)
    entity_precision = entity_correct / len(pred_entities) if pred_entities else 0
    entity_recall = entity_correct / len(gt_entities) if gt_entities else 0
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
    
    # 关系评估
    relation_correct = len(pred_relations & gt_relations)
    relation_precision = relation_correct / len(pred_relations) if pred_relations else 0
    relation_recall = relation_correct / len(gt_relations) if gt_relations else 0
    relation_f1 = 2 * relation_precision * relation_recall / (relation_precision + relation_recall) if (relation_precision + relation_recall) > 0 else 0
    
    return {
        "entity": {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1,
            "correct": entity_correct,
            "predicted": len(pred_entities),
            "ground_truth": len(gt_entities)
        },
        "relation": {
            "precision": relation_precision,
            "recall": relation_recall,
            "f1": relation_f1,
            "correct": relation_correct,
            "predicted": len(pred_relations),
            "ground_truth": len(gt_relations)
        }
    }


def evaluate_consistency(
    predicted_triples: List[Dict],
    schema_path: str
) -> Dict:
    """
    一致性评估：Schema一致性、对称性、互斥性
    """
    # 加载Schema
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_relations = json.load(f)
    
    schema_relation_names = {r["name"] for r in schema_relations}
    symmetric_relations = {r["name"] for r in schema_relations if r.get("is_symmetric", False)}
    mutex_pairs = []
    for r in schema_relations:
        if r.get("mutex_with"):
            for m in r["mutex_with"]:
                mutex_pairs.append((r["name"], m))
    
    # 1. Schema一致性：检查关系是否在Schema中
    schema_violations = []
    for t in predicted_triples:
        if t["relation"] not in schema_relation_names:
            schema_violations.append({
                "head": t["head"],
                "relation": t["relation"],
                "tail": t["tail"]
            })
    
    # 2. 对称性检查
    symmetric_violations = []
    relation_pairs = defaultdict(set)
    for t in predicted_triples:
        if t["relation"] in symmetric_relations:
            relation_pairs[t["relation"]].add((t["head"], t["tail"]))
    
    for relation, pairs in relation_pairs.items():
        for head, tail in pairs:
            if (tail, head) not in pairs:
                symmetric_violations.append({
                    "relation": relation,
                    "missing": f"{tail}-{relation}-{head}"
                })
    
    # 3. 互斥性检查
    entity_pair_relations = defaultdict(set)
    for t in predicted_triples:
        key = (t["head"], t["tail"])
        entity_pair_relations[key].add(t["relation"])
    
    mutex_violations = []
    for (head, tail), relations in entity_pair_relations.items():
        for r1, r2 in mutex_pairs:
            if r1 in relations and r2 in relations:
                mutex_violations.append({
                    "head": head,
                    "tail": tail,
                    "conflict_relations": [r1, r2]
                })
    
    return {
        "schema_violations": {
            "count": len(schema_violations),
            "examples": schema_violations[:5]
        },
        "symmetric_violations": {
            "count": len(symmetric_violations),
            "examples": symmetric_violations[:5]
        },
        "mutex_violations": {
            "count": len(mutex_violations),
            "examples": mutex_violations[:5]
        }
    }


def evaluate_completeness(
    predicted_triples: List[Dict],
    ground_truth: Dict,
    schema_path: str
) -> Dict:
    """
    完整性评估：实体覆盖率、关系覆盖率、知识密度
    """
    # 人工标注的答案 - entities是字符串数组
    entities_data = ground_truth["entities"]
    gt_entities = {normalize_entity(e) for e in entities_data}
    gt_relations = {r["relation"] for r in ground_truth["relations"]}
    
    # 预测实体
    pred_entities: Set[str] = set()
    for t in predicted_triples:
        pred_entities.add(normalize_entity(t["head"]))
        pred_entities.add(normalize_entity(t["tail"]))
    
    # 预测关系类型
    pred_relation_types = {t["relation"] for t in predicted_triples}
    
    # 实体覆盖率
    entity_coverage = len(pred_entities & gt_entities) / len(gt_entities) if gt_entities else 0
    
    # 关系覆盖率
    relation_coverage = len(pred_relation_types & gt_relations) / len(gt_relations) if gt_relations else 0
    
    # 知识密度（实体-关系比）
    entity_count = len(pred_entities)
    relation_count = len(predicted_triples)
    entity_relation_ratio = entity_count / relation_count if relation_count > 0 else 0
    
    # 实体连接度
    entity_connections = defaultdict(set)
    for t in predicted_triples:
        entity_connections[t["head"]].add(t["tail"])
        entity_connections[t["tail"]].add(t["head"])
    
    isolated_entities = [e for e in entity_connections if len(entity_connections[e]) == 0]
    avg_connectivity = sum(len(entity_connections[e]) for e in entity_connections) / len(entity_connections) if entity_connections else 0
    
    return {
        "entity_coverage": entity_coverage,
        "relation_coverage": relation_coverage,
        "entity_count": entity_count,
        "relation_count": relation_count,
        "entity_relation_ratio": entity_relation_ratio,
        "isolated_entities_count": len(isolated_entities),
        "isolated_entities_rate": len(isolated_entities) / entity_count if entity_count > 0 else 0,
        "avg_connectivity": avg_connectivity
    }


def run_on_eval(
    eval_dir: str,
    output_dir: str,
    schema_path: str,
    entity_types_path: str,
    api_key: str = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-v4-flash",
    embedding_model: str = None
) -> List[Dict]:
    """
    在eval.md上抽取(无需parse和preprocess)
    """
    raw_triples_path = f"{output_dir}/triples_raw.jsonl"
    normalized_triples_path = f"{output_dir}/triples_normalized.jsonl"
    final_triples_path = f"{output_dir}/triples_final.jsonl"
    normalization_log_path = f"{output_dir}/normalization.log"
    postprocess_log_path = f"{output_dir}/postprocess.log"

    print("\n[Step 1/3] 抽取知识三元组...")
    raw_count = run_extract(
        in_dir=eval_dir,
        out_jsonl=raw_triples_path,
        schema_path=schema_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
        chunk_chars=300,
        overlap=30,
    )
    
    # Step 2: 实体归一化和关系过滤
    print("\n[Step 2/3] 实体归一化和关系过滤...")
    normalized_count = run_normalize_and_filter(
        in_jsonl=raw_triples_path,
        out_jsonl=normalized_triples_path,
        model_name=embedding_model,
        schema_path=schema_path,
        log_path=normalization_log_path,
    )
    
    # Step 3: 后处理
    print("\n[Step 3/3] 后处理...")
    final_count = run_postprocess(
        in_jsonl=normalized_triples_path,
        out_jsonl=final_triples_path,
        entity_types_path=entity_types_path,
        relation_types_path=schema_path,
        embedding_model=embedding_model,
        log_path=postprocess_log_path,
    )
    
    # 加载最终结果
    triples = load_jsonl(final_triples_path)
    print(f"\n[PIPELINE DONE] 共抽取 {len(triples)} 个三元组")
    
    return triples


def generate_report(results: Dict, output_path: str) -> None:
    """生成评估报告"""
    ensure_parent(output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("综合评估报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体评分
        f.write("【一、总体评分】\n")
        f.write("-" * 80 + "\n")
        f.write(f"准确性得分:   {results['accuracy_score']:.2f}\n")
        f.write(f"一致性得分:   {results['consistency_score']:.2f}\n")
        f.write(f"完整性得分:   {results['completeness_score']:.2f}\n")
        f.write(f"综合得分:     {results['total_score']:.2f}\n\n")
        
        # 准确性详情
        f.write("【二、准确性评估】\n")
        f.write("-" * 80 + "\n")
        acc = results["accuracy"]
        f.write("实体评估:\n")
        f.write(f"  精确率: {acc['entity']['precision']:.2%}\n")
        f.write(f"  召回率: {acc['entity']['recall']:.2%}\n")
        f.write(f"  F1:     {acc['entity']['f1']:.2%}\n")
        f.write(f"  正确数: {acc['entity']['correct']}/{acc['entity']['predicted']}\n\n")
        
        f.write("关系评估:\n")
        f.write(f"  精确率: {acc['relation']['precision']:.2%}\n")
        f.write(f"  召回率: {acc['relation']['recall']:.2%}\n")
        f.write(f"  F1:     {acc['relation']['f1']:.2%}\n")
        f.write(f"  正确数: {acc['relation']['correct']}/{acc['relation']['predicted']}\n\n")
        
        # 一致性详情
        f.write("【三、一致性评估】\n")
        f.write("-" * 80 + "\n")
        cons = results["consistency"]
        f.write(f"Schema违规数:     {cons['schema_violations']['count']}\n")
        f.write(f"对称性违规数:     {cons['symmetric_violations']['count']}\n")
        f.write(f"互斥性违规数:     {cons['mutex_violations']['count']}\n\n")
        
        # 完整性详情
        f.write("【四、完整性评估】\n")
        f.write("-" * 80 + "\n")
        comp = results["completeness"]
        f.write(f"实体覆盖率:       {comp['entity_coverage']:.2%}\n")
        f.write(f"关系覆盖率:       {comp['relation_coverage']:.2%}\n")
        f.write(f"实体数量:         {comp['entity_count']}\n")
        f.write(f"关系数量:         {comp['relation_count']}\n")
        f.write(f"孤立实体比例:     {comp['isolated_entities_rate']:.2%}\n")
        f.write(f"平均连接度:       {comp['avg_connectivity']:.2f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("评估完成\n")
        f.write("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="综合评估工具")
    parser.add_argument("--eval-dir", default="data/evaluation")
    parser.add_argument("--ground-truth", default="data/evaluation/ground_truth.json")
    parser.add_argument("--output-dir", default="output/eval_result")
    parser.add_argument("--schema-path", default="config/relation_types.json")
    parser.add_argument("--entity-types-path", default="config/entity_types.json")
    parser.add_argument("--api-key", default=None, help="API密钥")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="API基础URL")
    parser.add_argument("--model", default="deepseek-v4-pro", help="模型名称")
    parser.add_argument("--embedding-model", default="model/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    print("=" * 80)
    print("开始评估流程")
    print("=" * 80)

    print("\n[1/4] 加载人工标注的答案...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"  人工标注的答案: {len(ground_truth['entities'])} 个实体, {len(ground_truth['relations'])} 个关系")

    print("\n[2/4] 在测试文档上运行pipeline...")
    predicted_triples = run_on_eval(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        schema_path=args.schema_path,
        entity_types_path=args.entity_types_path,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        embedding_model=args.embedding_model
    )
    
    # 准确性评估
    print("\n[3/4] 进行准确性评估...")
    accuracy_results = evaluate_accuracy(predicted_triples, ground_truth)
    
    # 一致性评估
    print("\n[4/4] 进行一致性评估...")
    consistency_results = evaluate_consistency(predicted_triples, args.schema_path)
    
    # 完整性评估
    print("\n[5/5] 进行完整性评估...")
    completeness_results = evaluate_completeness(predicted_triples, ground_truth, args.schema_path)
    
    # 计算得分
    # 准确性：实体F1 * 0.4 + 关系F1 * 0.6
    accuracy_score = (
        accuracy_results["entity"]["f1"] * 0.4 + 
        accuracy_results["relation"]["f1"] * 0.6
    ) * 100
    
    # 一致性：基于违规率（扣分制）
    total_triples = len(predicted_triples) if predicted_triples else 1
    violation_rate = (
        consistency_results["schema_violations"]["count"] +
        consistency_results["symmetric_violations"]["count"] +
        consistency_results["mutex_violations"]["count"]
    ) / total_triples
    consistency_score = max(0, (1 - violation_rate) * 100)
    
    # 完整性：实体覆盖率 * 0.4 + 关系覆盖率 * 0.4 + (1-孤立率) * 0.2
    completeness_score = (
        completeness_results["entity_coverage"] * 0.4 +
        completeness_results["relation_coverage"] * 0.4 +
        (1 - completeness_results["isolated_entities_rate"]) * 0.2
    ) * 100
    
    # 综合得分
    total_score = accuracy_score * 0.5 + consistency_score * 0.2 + completeness_score * 0.3
    
    # 汇总结果
    results = {
        "accuracy": accuracy_results,
        "consistency": consistency_results,
        "completeness": completeness_results,
        "accuracy_score": accuracy_score,
        "consistency_score": consistency_score,
        "completeness_score": completeness_score,
        "total_score": total_score
    }
    
    # 保存结果
    ensure_parent(f"{args.output_dir}/evaluation_results.json")
    with open(f"{args.output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成报告
    generate_report(results, f"{args.output_dir}/evaluation_report.txt")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)
    print(f"\n准确性得分:   {accuracy_score:.2f}")
    print(f"一致性得分:   {consistency_score:.2f}")
    print(f"完整性得分:   {completeness_score:.2f}")
    print(f"\n综合得分:     {total_score:.2f}")
    print(f"\n详细报告已保存至: {args.output_dir}/evaluation_report.txt")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()