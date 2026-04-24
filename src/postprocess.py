"""
知识图谱后处理模块
功能：
1. 实体类型标注：基于向量相似度为实体分配类型标签
2. 类型约束检查：验证三元组的头尾实体类型是否符合关系定义
3. 对称关系补全：自动补全对称关系的反向三元组
4. 互斥关系冲突解决：检测并解决互斥关系之间的冲突
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import dump_jsonl, load_jsonl


class YuanEmbeddingEncoder:
    """
    Yuan-Embedding向量编码器
    用于将文本转换为语义向量
    """
    
    def __init__(self, model_path: str):
        """加载模型和分词器"""
        print(f"加载Yuan-Embedding模型: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"模型已加载到: {self.device}")
    
    def encode(self, texts: List[str], normalize_embeddings: bool = True, show_progress_bar: bool = True) -> np.ndarray:
        """
        将文本编码为向量
        Args:
            texts: 文本列表
            normalize_embeddings: 是否归一化向量
            show_progress_bar: 是否显示进度条
        Returns:
            向量矩阵 (n, d)
        """
        from tqdm import tqdm
        
        embeddings = []
        batch_size = 32
        iterator = tqdm(range(0, len(texts), batch_size), desc="编码文本") if show_progress_bar else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                
                # Mean Pooling
                attention_mask = encoded_input["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                sentence_embeddings = sum_embeddings / sum_mask
            
            if normalize_embeddings:
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)


def load_entity_types(entity_types_path: str) -> List[Dict]:
    """
    加载实体类型配置
    Args:
        entity_types_path: entity_types.json文件路径
    Returns:
        实体类型列表
    """
    with open(entity_types_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_relation_types(relation_types_path: str) -> List[Dict]:
    """
    加载关系类型配置
    Args:
        relation_types_path: relation_types.json文件路径
    Returns:
        关系类型列表
    """
    with open(relation_types_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def annotate_entity_types(
    triples: List[Dict],
    entity_types: List[Dict],
    encoder: YuanEmbeddingEncoder,
    threshold: float = 0.6,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    为实体标注类型
    策略：
    1. 为每个实体类型构建原型向量（类型名称 + 描述 + 代表性实体拼接）
    2. 计算每个实体向量与所有类型原型向量的余弦相似度
    3. 取最高分且超过阈值的类型作为该实体的标签
    
    Args:
        triples: 三元组列表
        entity_types: 实体类型配置
        encoder: 向量编码器
        threshold: 类型标注阈值（默认0.6）
    Returns:
        (entity_to_type, entity_to_score): 实体到类型的映射和置信度分数
    """
    print("\n[实体类型标注] 开始...")
    
    # Step 1: 提取所有唯一实体
    entities = sorted({t["head"] for t in triples} | {t["tail"] for t in triples})
    print(f"  待标注实体数: {len(entities)}")
    
    # Step 2: 构建类型原型文本
    # 策略：每个类型生成多个原型（类型名本身 + 每个示例实体单独编码），
    # 取实体与该类型所有原型的最大相似度作为匹配分数，
    # 这样短实体词能与短示例词直接比较，避免长描述文本带来的语义偏差
    type_names = [et["name"] for et in entity_types]
    # 每个类型的原型文本列表：类型名 + 所有示例
    type_prototype_groups = []
    for et in entity_types:
        prototypes = [et["name"]] + et.get("examples", [])
        type_prototype_groups.append(prototypes)
    
    # 展平所有原型，记录每个原型属于哪个类型
    all_prototypes = []
    prototype_type_idx = []  # 每个原型对应的类型索引
    for type_idx, prototypes in enumerate(type_prototype_groups):
        for p in prototypes:
            all_prototypes.append(p)
            prototype_type_idx.append(type_idx)
    
    print(f"  实体类型数: {len(type_names)}，原型总数: {len(all_prototypes)}")
    
    # Step 3: 编码所有原型向量
    print("  编码类型原型向量...")
    prototype_embeddings = encoder.encode(all_prototypes, normalize_embeddings=True, show_progress_bar=False)
    
    # Step 4: 编码实体向量
    print("  编码实体向量...")
    entity_embeddings = encoder.encode(entities, normalize_embeddings=True, show_progress_bar=True)
    
    # Step 5: 计算相似度并分配类型
    print("  计算相似度并分配类型...")
    entity_to_type = {}
    entity_to_score = {}
    untyped_count = 0
    
    # 计算实体与所有原型的相似度矩阵：(n_entities, n_prototypes)
    similarity_matrix = np.dot(entity_embeddings, prototype_embeddings.T)
    
    for i, entity in enumerate(entities):
        # 对每个类型，取该类型所有原型中的最大相似度
        type_scores = np.zeros(len(type_names))
        for proto_idx, type_idx in enumerate(prototype_type_idx):
            score = similarity_matrix[i, proto_idx]
            if score > type_scores[type_idx]:
                type_scores[type_idx] = score
        
        max_type_idx = np.argmax(type_scores)
        max_score = type_scores[max_type_idx]
        
        if max_score >= threshold:
            entity_to_type[entity] = type_names[max_type_idx]
            entity_to_score[entity] = float(max_score)
        else:
            untyped_count += 1
    
    print(f"  已标注实体: {len(entity_to_type)} 个")
    print(f"  未标注实体: {untyped_count} 个（相似度低于阈值 {threshold}）")
    
    # 统计各类型的实体数量
    type_counts = {}
    for entity_type in entity_to_type.values():
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    print("  类型分布:")
    for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {type_name}: {count} 个")
    
    return entity_to_type, entity_to_score


def check_type_constraints(
    triples: List[Dict],
    entity_to_type: Dict[str, str],
    relation_types: List[Dict],
    log_file=None,
) -> List[Dict]:
    """
    检查类型约束
    只有实体有类型标注且类型不在允许集合内时才过滤，未标注类型的实体直接放行
    Args:
        triples: 三元组列表
        entity_to_type: 实体到类型的映射
        relation_types: 关系类型配置（包含head_types和tail_types字段）
        log_file: 日志文件对象
    Returns:
        通过类型约束检查的三元组列表
    """
    print("\n[类型约束检查] 开始...")

    def log(msg):
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    relation_constraints = {}
    for rt in relation_types:
        relation_name = rt["name"]
        head_types = set(rt.get("head_types", []))
        tail_types = set(rt.get("tail_types", []))
        if head_types or tail_types:
            relation_constraints[relation_name] = {
                "head_types": head_types,
                "tail_types": tail_types,
            }
    
    if not relation_constraints:
        log("  警告: relation_types.json中未定义类型约束，跳过检查")
        return triples
    
    log(f"  关系类型约束数: {len(relation_constraints)}")
    
    valid_triples = []
    invalid_count = 0
    # 记录被过滤的原因分布
    filter_reasons: Dict[str, int] = {}
    
    for triple in triples:
        head = triple["head"]
        relation = triple["relation"]
        tail = triple["tail"]
        
        if relation not in relation_constraints:
            valid_triples.append(triple)
            continue
        
        constraints = relation_constraints[relation]
        head_type = entity_to_type.get(head)
        tail_type = entity_to_type.get(tail)
        
        head_valid = True
        if constraints["head_types"] and head_type:
            if head_type not in constraints["head_types"]:
                head_valid = False
        
        tail_valid = True
        if constraints["tail_types"] and tail_type:
            if tail_type not in constraints["tail_types"]:
                tail_valid = False
        
        if head_valid and tail_valid:
            valid_triples.append(triple)
        else:
            invalid_count += 1
            reason = f"{relation}: head={head_type or '未标注'}, tail={tail_type or '未标注'}"
            filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
            if log_file:
                log_file.write(f"  过滤: ({head})-[{relation}]->({tail}) | head_type={head_type or '未标注'}, tail_type={tail_type or '未标注'}\n")
    
    log(f"  通过检查: {len(valid_triples)} 条")
    log(f"  未通过检查: {invalid_count} 条")
    
    if filter_reasons and log_file:
        log_file.write("\n  过滤原因统计（按频次降序）:\n")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1])[:30]:
            log_file.write(f"    [{count}次] {reason}\n")
    
    return valid_triples


def complete_symmetric_relations(
    triples: List[Dict],
    relation_types: List[Dict],
) -> List[Dict]:
    """
    补全对称关系
    对于标记为对称的关系，自动生成反向三元组
    
    Args:
        triples: 三元组列表
        relation_types: 关系类型配置（包含is_symmetric字段）
    Returns:
        补全后的三元组列表
    """
    print("\n[对称关系补全] 开始...")
    
    # 找出所有对称关系
    symmetric_relations = set()
    for rt in relation_types:
        if rt.get("is_symmetric", False):
            symmetric_relations.add(rt["name"])
    
    if not symmetric_relations:
        print("  未定义对称关系，跳过补全")
        return triples
    
    print(f"  对称关系: {symmetric_relations}")
    
    # 构建现有三元组集合（用于去重）
    existing_triples = {(t["head"], t["relation"], t["tail"]) for t in triples}
    
    # 补全对称关系
    completed_triples = list(triples)
    added_count = 0
    
    for triple in triples:
        relation = triple["relation"]
        
        # 只处理对称关系
        if relation not in symmetric_relations:
            continue
        
        # 生成反向三元组
        reverse_key = (triple["tail"], relation, triple["head"])
        
        # 如果反向三元组不存在，则添加
        if reverse_key not in existing_triples:
            completed_triples.append({
                "head": triple["tail"],
                "relation": relation,
                "tail": triple["head"],
                "source": triple.get("source", ""),
                "method": "symmetric_completion",
            })
            existing_triples.add(reverse_key)
            added_count += 1
    
    print(f"  补全三元组: {added_count} 条")
    print(f"  总三元组数: {len(completed_triples)} 条")
    
    return completed_triples


def resolve_mutex_relations(
    triples: List[Dict],
    relation_types: List[Dict],
) -> List[Dict]:
    """
    解决互斥关系冲突
    检测互斥关系之间的冲突，保留置信度更高或来源更可靠的三元组
    
    Args:
        triples: 三元组列表
        relation_types: 关系类型配置（包含mutex_with字段）
    Returns:
        解决冲突后的三元组列表
    """
    print("\n[互斥关系冲突解决] 开始...")
    
    # 构建互斥关系映射
    mutex_map: Dict[str, Set[str]] = {}
    for rt in relation_types:
        relation_name = rt["name"]
        mutex_with = rt.get("mutex_with", [])
        if mutex_with:
            mutex_map[relation_name] = set(mutex_with)
    
    if not mutex_map:
        print("  未定义互斥关系，跳过冲突解决")
        return triples
    
    print(f"  互斥关系对数: {sum(len(v) for v in mutex_map.values()) // 2}")
    
    # 按(head, tail)分组三元组
    entity_pair_triples: Dict[Tuple[str, str], List[Dict]] = {}
    for triple in triples:
        key = (triple["head"], triple["tail"])
        if key not in entity_pair_triples:
            entity_pair_triples[key] = []
        entity_pair_triples[key].append(triple)
    
    # 检测冲突并解决
    resolved_triples = []
    conflict_count = 0
    
    for (head, tail), group in entity_pair_triples.items():
        # 检查是否存在互斥关系
        relations_in_group = [t["relation"] for t in group]
        
        has_conflict = False
        for i, r1 in enumerate(relations_in_group):
            for r2 in relations_in_group[i+1:]:
                if r1 in mutex_map and r2 in mutex_map[r1]:
                    has_conflict = True
                    break
            if has_conflict:
                break
        
        if not has_conflict:
            # 无冲突，保留所有三元组
            resolved_triples.extend(group)
        else:
            # 有冲突，选择保留策略：
            # 1. 优先保留method="normalized"的（来自归一化阶段）
            # 2. 其次按关系名称字典序（确保一致性）
            conflict_count += len(group) - 1
            
            # 排序：normalized优先，然后按关系名
            sorted_group = sorted(group, key=lambda t: (
                0 if t.get("method") == "normalized" else 1,
                t["relation"]
            ))
            
            # 只保留第一个
            resolved_triples.append(sorted_group[0])
    
    print(f"  检测到冲突: {conflict_count} 条")
    print(f"  解决后三元组: {len(resolved_triples)} 条")
    
    return resolved_triples


def run_postprocess(
    in_jsonl: str,
    out_jsonl: str,
    entity_types_path: str = "config/entity_types.json",
    relation_types_path: str = "config/relation_types.json",
    embedding_model: str = "model/Yuan-Embedding",
    type_threshold: float = 0.6,
    enable_type_annotation: bool = True,
    enable_type_check: bool = True,
    enable_symmetric_completion: bool = True,
    enable_mutex_resolution: bool = True,
    log_path: str = None,
) -> int:
    """
    运行后处理流程
    Args:
        in_jsonl: 输入文件路径（归一化后的三元组）
        out_jsonl: 输出文件路径（后处理后的三元组）
        entity_types_path: 实体类型配置文件路径
        relation_types_path: 关系类型配置文件路径
        embedding_model: Yuan-Embedding模型路径
        type_threshold: 实体类型标注阈值
        enable_type_annotation: 是否启用实体类型标注
        enable_type_check: 是否启用类型约束检查
        enable_symmetric_completion: 是否启用对称关系补全
        enable_mutex_resolution: 是否启用互斥关系冲突解决
        log_path: 日志文件路径
    Returns:
        后处理后的三元组数量
    """
    print("="*60)
    print("知识图谱后处理")
    print("="*60)
    
    triples = load_jsonl(in_jsonl)
    if not triples:
        raise ValueError("输入三元组为空")
    
    print(f"\n加载三元组: {len(triples)} 条")
    
    entity_types = load_entity_types(entity_types_path)
    relation_types = load_relation_types(relation_types_path)
    
    print(f"加载实体类型: {len(entity_types)} 个")
    print(f"加载关系类型: {len(relation_types)} 个")

    # 准备日志文件
    log_file = None
    if log_path:
        from pathlib import Path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, 'w', encoding='utf-8')
        log_file.write("知识图谱后处理日志\n")
        log_file.write("="*60 + "\n\n")
    
    try:
        # Step 1: 实体类型标注
        entity_to_type = {}
        entity_to_score = {}
        if enable_type_annotation:
            encoder = YuanEmbeddingEncoder(embedding_model)
            entity_to_type, entity_to_score = annotate_entity_types(
                triples, entity_types, encoder, threshold=type_threshold
            )
            
            # 将类型信息写入日志
            if log_file:
                log_file.write("[实体类型标注结果]\n")
                log_file.write(f"已标注: {len(entity_to_type)} 个\n")
                log_file.write(f"未标注: {len({t['head'] for t in triples} | {t['tail'] for t in triples}) - len(entity_to_type)} 个\n\n")
                log_file.write("实体 → 类型 (置信度)\n")
                for entity, etype in sorted(entity_to_type.items()):
                    score = entity_to_score.get(entity, 0.0)
                    log_file.write(f"  {entity} → {etype} ({score:.3f})\n")
                log_file.write("\n")
            
            for triple in triples:
                triple["head_type"] = entity_to_type.get(triple["head"], "")
                triple["tail_type"] = entity_to_type.get(triple["tail"], "")
                triple["head_type_score"] = entity_to_score.get(triple["head"], 0.0)
                triple["tail_type_score"] = entity_to_score.get(triple["tail"], 0.0)
        
        # Step 2: 类型约束检查
        if enable_type_check and entity_to_type:
            if log_file:
                log_file.write("="*60 + "\n[类型约束检查 - 被过滤的三元组]\n")
            triples = check_type_constraints(triples, entity_to_type, relation_types, log_file=log_file)
        
        # Step 3: 对称关系补全
        if enable_symmetric_completion:
            triples = complete_symmetric_relations(triples, relation_types)
        
        # Step 4: 互斥关系冲突解决
        if enable_mutex_resolution:
            triples = resolve_mutex_relations(triples, relation_types)
        
        dump_jsonl(out_jsonl, triples)
        
        entity_count = len({t["head"] for t in triples} | {t["tail"] for t in triples})
        relation_count = len({t["relation"] for t in triples})
        
        print("\n" + "="*60)
        print("[DONE] 后处理完成")
        print(f"  - 最终三元组: {len(triples)} 条")
        print(f"  - 唯一实体: {entity_count} 个")
        print(f"  - 唯一关系: {relation_count} 个")
        print(f"[SAVE] {out_jsonl}")
        if log_path:
            print(f"[日志] 后处理详情已写入: {log_path}")
        print("="*60)
        
    finally:
        if log_file:
            log_file.close()
    
    return len(triples)


def main() -> None:
    """
    使用示例：
    python postprocess.py \
        --in-jsonl ./output/triples_normalized/triples.jsonl \
        --out-jsonl ./output/triples_final/triples.jsonl \
        --type-threshold 0.6
    """
    parser = argparse.ArgumentParser(description="知识图谱后处理")
    parser.add_argument("--in-jsonl", required=True, help="输入JSONL文件路径（归一化后的三元组）")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径（后处理后的三元组）")
    parser.add_argument("--entity-types", default="config/entity_types.json", help="实体类型配置文件路径")
    parser.add_argument("--relation-types", default="config/relation_types.json", help="关系类型配置文件路径")
    parser.add_argument("--embedding-model", default="model/Yuan-Embedding", help="Yuan-Embedding模型路径")
    parser.add_argument("--type-threshold", type=float, default=0.6, help="实体类型标注阈值")
    parser.add_argument("--disable-type-annotation", action="store_true", help="禁用实体类型标注")
    parser.add_argument("--disable-type-check", action="store_true", help="禁用类型约束检查")
    parser.add_argument("--disable-symmetric-completion", action="store_true", help="禁用对称关系补全")
    parser.add_argument("--disable-mutex-resolution", action="store_true", help="禁用互斥关系冲突解决")
    parser.add_argument("--log-path", default="output/logs/postprocess.log", help="日志文件路径")
    args = parser.parse_args()

    run_postprocess(
        in_jsonl=args.in_jsonl,
        out_jsonl=args.out_jsonl,
        entity_types_path=args.entity_types,
        relation_types_path=args.relation_types,
        embedding_model=args.embedding_model,
        type_threshold=args.type_threshold,
        enable_type_annotation=not args.disable_type_annotation,
        enable_type_check=not args.disable_type_check,
        enable_symmetric_completion=not args.disable_symmetric_completion,
        enable_mutex_resolution=not args.disable_mutex_resolution,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
