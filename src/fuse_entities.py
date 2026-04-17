"""
使用Yuan-Embedding向量模型进行实体与关系归一化融合
功能：
1. 加载OneKE抽取的三元组
2. 使用Yuan-Embedding中文向量模型计算实体和关系的语义向量
3. 基于余弦相似度聚类，合并同义实体
4. 合并近义关系
5. 输出归一化、去重后的标准化三元组
"""

import argparse
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from common import dump_jsonl, load_jsonl


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


def build_clusters(items: List[str], embeddings: np.ndarray, threshold: float) -> Dict[str, str]:
    """
    基于向量相似度对文本进行贪心聚类，返回每个文本到标准名的映射
    流程：
    1. 遍历所有未访问的文本
    2. 将当前文本与所有未访问文本比较相似度
    3. 相似度超过阈值的归为同一簇
    4. 选择簇中最短字符串作为标准名（偏向简洁命名）
    5. 继续处理下一个未访问文本
    Args:
        items: 文本列表，如["变构飞行器", "变体飞行器", "可变构飞行器"]
        embeddings: 对应的向量矩阵
        threshold: 相似度阈值，大于等于此值视为同一语义
    Returns:
        映射字典，如{"变构飞行器": "变构飞行器", "变体飞行器": "变构飞行器"}
    """

    # 计算相似度矩阵
    sim = cosine_sim_matrix(embeddings)
    n = len(items)
    visited = [False] * n      # 标记哪些文本已被归类
    canonical = {}              # 存储最终映射关系
    
    for i in range(n):
        if visited[i]:
            continue
        
        # 开始新簇，包含当前文本
        visited[i] = True
        cluster = [i]
        
        # 寻找所有与i相似的文本
        for j in range(i + 1, n):
            if visited[j]:
                continue
            if sim[i, j] >= threshold:
                visited[j] = True
                cluster.append(j)
        
        # 选择簇中最短的字符串作为标准名
        name = sorted([items[k] for k in cluster], key=lambda x: (len(x), x))[0]
        for k in cluster:
            canonical[items[k]] = name
    
    return canonical


def normalize(
    triples: List[Dict],
    model_name: str = "model/Yuan-Embedding",
    entity_threshold: float = 0.85,
    relation_threshold: float = 0.9,
) -> List[Dict]:
    """
    使用Yuan-Embedding向量模型对三元组进行实体和关系的归一化融合
    流程：
    1. 提取所有唯一的实体（头实体+尾实体）和关系
    2. 用Yuan-Embedding模型将文本转换为向量
    3. 基于相似度聚类，合并同义实体和近义关系
    4. 应用映射，生成归一化后的三元组
    5. 去重后返回
    Args:
        triples: 原始三元组列表，每个元素包含head、relation、tail字段
        model_name: Yuan-Embedding模型路径，默认使用本地model/Yuan-Embedding
        entity_threshold: 实体相似度阈值（0.85），高于此值视为同一实体
        relation_threshold: 关系相似度阈值（0.9），关系要求更严格
    Returns:
        归一化、去重后的三元组列表
    """

    # Step 1: 加载向量模型
    print(f"加载Yuan-Embedding模型: {model_name}")
    model = SentenceTransformer(model_name)

    # Step 2: 提取所有唯一的实体和关系
    entities = sorted({t["head"].strip() for t in triples} | {t["tail"].strip() for t in triples})
    relations = sorted({t["relation"].strip() for t in triples})
    
    print(f"原始统计: {len(entities)} 个唯一实体, {len(relations)} 个唯一关系")

    # Step 3: 生成向量
    entity_emb = model.encode(entities, normalize_embeddings=True, show_progress_bar=True)
    relation_emb = model.encode(relations, normalize_embeddings=True, show_progress_bar=False)

    # Step 4: 聚类合并
    entity_map = build_clusters(entities, entity_emb, threshold=entity_threshold)
    relation_map = build_clusters(relations, relation_emb, threshold=relation_threshold)
    
    # 打印映射统计
    unique_canonical_entities = len(set(entity_map.values()))
    unique_canonical_relations = len(set(relation_map.values()))
    print(f"实体归一化: {len(entities)} → {unique_canonical_entities} 个标准实体")
    print(f"关系归一化: {len(relations)} → {unique_canonical_relations} 个标准关系")

    # Step 5: 应用映射并去重
    fused: List[Dict] = []
    dedup = set()
    
    for t in triples:
        # 获取归一化后的值，如果不在映射中则保留原值
        h = entity_map.get(t["head"].strip(), t["head"].strip())
        r = relation_map.get(t["relation"].strip(), t["relation"].strip())
        ta = entity_map.get(t["tail"].strip(), t["tail"].strip())
        
        # 去重
        key = (h, r, ta)
        if key in dedup:
            continue
        dedup.add(key)
        
        fused.append(
            {
                "head": h,
                "relation": r,
                "tail": ta,
                "source": t.get("source", ""),
                "method": "bge_normalization",  # 标记经过融合处理
            }
        )
    
    return fused


def run_fusion(
    in_jsonl: str,
    out_jsonl: str,
    model_name: str = "model/Yuan-Embedding",
    entity_threshold: float = 0.85,
    relation_threshold: float = 0.9,
) -> int:
    """
    运行Yuan-Embedding知识融合的主函数
    Args:
        in_jsonl: 输入文件路径（OneKE抽取的三元组，JSONL格式）
        out_jsonl: 输出文件路径（融合后的三元组）
        model_name: Yuan-Embedding模型路径
        entity_threshold: 实体相似度阈值
        relation_threshold: 关系相似度阈值
    Returns:
        融合后的三元组数量
    """

    # 加载原始三元组
    triples = load_jsonl(in_jsonl)
    if not triples:
        raise ValueError("输入三元组为空，无法融合。")
    
    print(f"加载原始三元组: {len(triples)} 条")

    # 执行归一化融合
    fused = normalize(
        triples,
        model_name=model_name,
        entity_threshold=entity_threshold,
        relation_threshold=relation_threshold,
    )
    
    # 保存结果
    dump_jsonl(out_jsonl, fused)
    
    # 统计信息
    entity_count = len({x["head"] for x in fused} | {x["tail"] for x in fused})
    print(f"[DONE] Yuan-Embedding融合完成:")
    print(f"  - 融合后三元组: {len(fused)} 条")
    print(f"  - 融合后唯一实体: {entity_count} 个")
    print(f"  - 压缩率: {(1 - len(fused)/len(triples)) * 100:.1f}%")
    print(f"[SAVE] {out_jsonl}")
    
    return len(fused)


def main() -> None:
    """
    使用示例：
    python fuse_entities.py \
        --in-jsonl ./triples.jsonl \
        --out-jsonl ./triples_fused.jsonl \
        --entity-threshold 0.85 \
        --relation-threshold 0.9
    """

    parser = argparse.ArgumentParser(description="使用Yuan-Embedding向量模型融合实体和关系")
    parser.add_argument("--in-jsonl", required=True, help="输入JSONL文件路径（OneKE抽取结果）")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径（融合后结果）")
    parser.add_argument("--embedding-model", default="model/Yuan-Embedding", help="Yuan-Embedding模型路径")
    parser.add_argument("--entity-threshold", type=float, default=0.85, 
                        help="实体相似度阈值（0.7-0.9），越高越严格，越低越激进")
    parser.add_argument("--relation-threshold", type=float, default=0.9,
                        help="关系相似度阈值（0.8-0.95），关系要求比实体更严格")
    args = parser.parse_args()

    run_fusion(
        in_jsonl=args.in_jsonl,
        out_jsonl=args.out_jsonl,
        model_name=args.embedding_model,
        entity_threshold=args.entity_threshold,
        relation_threshold=args.relation_threshold,
    )


if __name__ == "__main__":
    main()