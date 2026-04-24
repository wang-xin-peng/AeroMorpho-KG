"""
使用向量模型进行实体归一化和关系过滤
向量模型：Qwen3-Embedding-0.6B
功能：
1. 加载OneKE抽取的原始三元组
2. 使用向量模型计算实体的语义向量
3. 基于余弦相似度 complete-linkage 聚类，合并同义实体
4. 根据预定义schema过滤无效关系
5. 输出归一化、去重后的三元组
"""

import argparse
import json
import os
from typing import Dict, List
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from common import dump_jsonl, load_jsonl

class EmbeddingEncoder:
    """通用向量编码器，使用 mean pooling"""

    def __init__(self, model_path: str):
        print(f"加载向量模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.model_path = model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right", local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, local_files_only=True)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"模型架构: mean pooling，设备: {self.device}")

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def encode(self, texts: List[str], normalize_embeddings: bool = True,
               show_progress_bar: bool = True) -> np.ndarray:
        """
        将文本编码为向量，使用 mean pooling
        """
        from tqdm import tqdm

        embeddings = []
        batch_size = 32
        iterator = (tqdm(range(0, len(texts), batch_size), desc="编码文本")
                    if show_progress_bar else range(0, len(texts), batch_size))

        for i in iterator:
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                token_emb = outputs.last_hidden_state.float()
                attn_mask = encoded["attention_mask"]
                sent_emb = self._mean_pool(token_emb, attn_mask)

            if normalize_embeddings:
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

            embeddings.append(sent_emb.cpu().numpy())

        return np.vstack(embeddings)


def cosine_sim_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    计算向量之间的余弦相似度矩阵。
    余弦相似度公式
    Args:
        embeddings: 形状为(n, d)的向量矩阵
    Returns:
        形状为(n, n)的相似度矩阵，sim[i][j]表示第i个和第j个向量的余弦相似度
    """

    # 向量已归一化时，点积等于余弦相似度
    return np.dot(embeddings, embeddings.T)



def build_clusters_with_scores(items: List[str], embeddings: np.ndarray, threshold: float) -> tuple:
    """
    聚类算法：single-linkage + 后验簇内验证。
    相比纯 complete-linkage，速度快 100 倍，质量接近。
    
    流程：
    1. 用 single-linkage 快速聚类（O(n²)）
    2. 对每个簇进行后验检查：计算簇内最小相似度
    3. 若簇内最小相似度 < threshold，拆分该簇（保守策略）
    
    Returns:
        (canonical_map, merge_scores):
            canonical_map  - {原始实体: 标准名}
            merge_scores   - {原始实体: (标准名, 与标准名的相似度)}
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    
    sim = cosine_sim_matrix(embeddings)
    n = len(items)
    
    # Step 1: 转换为距离矩阵（1 - 相似度），clip 防止浮点误差导致负值
    distance_matrix = np.clip(1 - sim, 0.0, 2.0)
    # 只取上三角（scipy 要求）
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Step 2: single-linkage 层次聚类
    linkage_matrix = linkage(condensed_dist, method='single')
    
    # Step 3: 按阈值切分
    distance_threshold = 1 - threshold
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    
    # Step 4: 按簇分组
    clusters_dict: Dict[int, List[int]] = {}
    for idx, label in enumerate(cluster_labels):
        clusters_dict.setdefault(label, []).append(idx)
    
    # Step 5: 后验验证 - 拆分不满足 complete-linkage 的簇
    validated_clusters = []
    for cluster_indices in clusters_dict.values():
        if len(cluster_indices) == 1:
            validated_clusters.append(cluster_indices)
            continue
        
        # 计算簇内最小相似度
        min_sim = 1.0
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                min_sim = min(min_sim, sim[idx_i, idx_j])
        
        # 若簇内最小相似度 >= threshold，保留整个簇
        if min_sim >= threshold:
            validated_clusters.append(cluster_indices)
        else:
            # 否则拆分：每个成员自成一簇（保守策略）
            for idx in cluster_indices:
                validated_clusters.append([idx])
    
    # Step 6: 为每个簇选标准名
    canonical: Dict[str, str] = {}
    merge_scores: Dict[str, tuple] = {}
    
    for cluster in validated_clusters:
        name = sorted([items[k] for k in cluster], key=lambda x: (len(x), x))[0]
        name_idx = items.index(name)
        for k in cluster:
            canonical[items[k]] = name
            if items[k] != name:
                merge_scores[items[k]] = (name, float(sim[k, name_idx]))
    
    return canonical, merge_scores


def normalize_and_filter(
    triples: List[Dict],
    model_name: str = "model/Qwen3-Embedding-0.6B",
    entity_threshold: float = 0.93,
    schema_path: str = "config/relation_types.json",
    log_path: str = None,
) -> List[Dict]:
    """
    对三元组进行实体归一化和关系过滤
    流程：
    1. 加载schema中的预定义关系
    2. 提取所有唯一的实体（头实体+尾实体）
    3. 用向量模型将实体转换为向量
    4. 基于 complete-linkage 相似度聚类，合并同义实体
    5. 应用实体映射，过滤无效关系
    6. 去重后返回
    Args:
        triples: 原始三元组列表，每个元素包含head、relation、tail字段
        model_name: 向量模型路径，默认使用本地 model/Qwen3-Embedding-0.6B
        entity_threshold: 实体相似度阈值（0.93），高于此值视为同一实体
        schema_path: schema配置文件路径，用于验证关系有效性
        log_path: 日志文件路径，记录归一化详情
    Returns:
        归一化、去重后的三元组列表
    """
        
    # Step 1: 加载schema中的预定义关系
    valid_relations = set()
    if Path(schema_path).exists():
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
            valid_relations = {item["name"] for item in schema_data}
        print(f"加载schema: {len(valid_relations)} 个预定义关系")
    else:
        print(f"警告: schema文件不存在 {schema_path}，跳过关系验证")

    # Step 2: 加载向量模型
    encoder = EmbeddingEncoder(model_name)

    # Step 3: 提取所有唯一的实体
    entities = sorted({t["head"].strip() for t in triples} | {t["tail"].strip() for t in triples})
    relations = sorted({t["relation"].strip() for t in triples})
    
    print(f"原始统计: {len(entities)} 个唯一实体, {len(relations)} 个唯一关系")

    # Step 4: 对实体生成向量并聚类
    print("编码实体向量...")
    entity_emb = encoder.encode(entities, normalize_embeddings=True, show_progress_bar=True)

    # Step 5: 对实体进行聚类合并
    entity_map, merge_scores = build_clusters_with_scores(entities, entity_emb, threshold=entity_threshold)
    
    # 打印归一化统计
    unique_canonical_entities = len(set(entity_map.values()))
    print(f"实体归一化: {len(entities)} → {unique_canonical_entities} 个标准实体")

    # Step 6: 应用实体映射，过滤无效关系，去重
    normalized: List[Dict] = []
    dedup = set()
    invalid_relations = set()
    
    for t in triples:
        h = entity_map.get(t["head"].strip(), t["head"].strip())
        r = t["relation"].strip()
        ta = entity_map.get(t["tail"].strip(), t["tail"].strip())
        
        if valid_relations and r not in valid_relations:
            invalid_relations.add(r)
            continue
        
        key = (h, r, ta)
        if key in dedup:
            continue
        dedup.add(key)
        
        normalized.append(
            {
                "head": h,
                "relation": r,
                "tail": ta,
                "source": t.get("source", ""),
                "method": "normalized",
            }
        )
    
    if invalid_relations:
        print(f"关系过滤：过滤了 {len(invalid_relations)} 个无效关系: {invalid_relations}")

    # 写日志
    if log_path:
        _write_normalization_log(log_path, entities, entity_map, merge_scores, entity_threshold, invalid_relations)
        print(f"[日志] 归一化详情已写入: {log_path}")
    
    return normalized


def _write_normalization_log(
    log_path: str,
    entities: List[str],
    entity_map: Dict[str, str],
    merge_scores: Dict[str, tuple],
    threshold: float,
    invalid_relations: set,
) -> None:
    """写实体归一化日志，记录每个合并簇的详情及相似度分数"""
    from pathlib import Path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # 按标准名分组
    clusters: Dict[str, List[str]] = {}
    for original, canonical in entity_map.items():
        clusters.setdefault(canonical, []).append(original)

    merged = {k: v for k, v in clusters.items() if len(v) > 1}
    singleton = len(clusters) - len(merged)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"实体归一化日志\n")
        f.write(f"{'='*60}\n")
        f.write(f"阈值: {threshold}\n")
        f.write(f"原始实体数: {len(entities)}\n")
        f.write(f"标准实体数: {len(clusters)}\n")
        f.write(f"发生合并的簇: {len(merged)} 个\n")
        f.write(f"未合并的实体: {singleton} 个\n")
        if invalid_relations:
            f.write(f"过滤的无效关系: {invalid_relations}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"合并详情（按簇大小降序，括号内为与簇中心的相似度）\n")
        f.write(f"{'='*60}\n\n")

        for canonical, members in sorted(merged.items(), key=lambda x: -len(x[1])):
            others = [m for m in members if m != canonical]
            f.write(f"标准名: 【{canonical}】 (共 {len(members)} 个)\n")
            for m in sorted(others):
                score = merge_scores.get(m, (None, 0.0))[1]
                f.write(f"  ← {m}  (sim={score:.4f})\n")
            f.write("\n")


def run_normalize_and_filter(
    in_jsonl: str,
    out_jsonl: str,
    model_name: str = "model/Qwen3-Embedding-0.6B",
    entity_threshold: float = 0.93,
    schema_path: str = "config/relation_types.json",
    log_path: str = None,
) -> int:
    """
    运行实体归一化和关系过滤的主函数
    Args:
        in_jsonl: 输入文件路径
        out_jsonl: 输出文件路径
        model_name: 向量模型路径
        entity_threshold: 实体相似度阈值
        schema_path: schema配置文件路径
        log_path: 日志文件路径（记录归一化合并详情）
    Returns:
        归一化后的三元组数量
    """
    print("="*60)
    print("归一化")
    print("="*60)

    triples = load_jsonl(in_jsonl)
    if not triples:
        raise ValueError("输入三元组为空，无法处理。")
    
    print(f"加载原始三元组: {len(triples)} 条")

    normalized = normalize_and_filter(
        triples,
        model_name=model_name,
        entity_threshold=entity_threshold,
        schema_path=schema_path,
        log_path=log_path,
    )
    
    dump_jsonl(out_jsonl, normalized)
    
    entity_count = len({x["head"] for x in normalized} | {x["tail"] for x in normalized})
    print(f"[DONE] 归一化和过滤完成:")
    print(f"  - 归一化后三元组: {len(normalized)} 条")
    print(f"  - 归一化后唯一实体: {entity_count} 个")
    print(f"  - 压缩率: {(1 - len(normalized)/len(triples)) * 100:.1f}%")
    print(f"[SAVE] {out_jsonl}")
    
    return len(normalized)


def main() -> None:
    """
    使用示例：
    python normalize_and_filter.py \
        --in-jsonl ./output/triples_raw/triples.jsonl \
        --out-jsonl ./output/triples_normalized/triples.jsonl \
    """

    parser = argparse.ArgumentParser(description="实体归一化和关系过滤")
    parser.add_argument("--in-jsonl", required=True, help="输入JSONL文件路径（OneKE抽取结果）")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径（归一化后结果）")
    parser.add_argument("--embedding-model", default="model/Qwen3-Embedding-0.6B", help="向量模型路径，默认Qwen3-Embedding")
    parser.add_argument("--schema-path", default="config/relation_types.json", help="schema配置文件路径")
    parser.add_argument("--entity-threshold", type=float, default=0.93, help="实体相似度阈值")
    parser.add_argument("--log-path", default="output/logs/normalization.log", help="归一化日志文件路径")
    args = parser.parse_args()

    run_normalize_and_filter(
        in_jsonl=args.in_jsonl,
        out_jsonl=args.out_jsonl,
        model_name=args.embedding_model,
        entity_threshold=args.entity_threshold,
        schema_path=args.schema_path,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()