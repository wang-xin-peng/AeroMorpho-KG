"""
使用向量模型进行实体归一化和关系过滤
支持模型：
  - BAAI/bge-large-zh-v1.5  （mean pooling，BERT 架构）
  - Qwen3-Embedding-0.6B    （last token pooling，Decoder 架构，推荐）
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

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import dump_jsonl, load_jsonl

# Qwen3-Embedding 官方推荐的实体归一化任务指令
QWEN3_TASK_INSTRUCTION = "找出与以下航空航天专业术语语义相同的实体"


def _is_decoder_model(model_path: str) -> bool:
    """通过 config.json 判断是否为 Decoder-only 架构（如 Qwen3-Embedding）"""
    config_file = os.path.join(model_path, "config.json")
    if not os.path.exists(config_file):
        return False
    with open(config_file, encoding="utf-8") as f:
        cfg = json.load(f)
    # Qwen3 / LLaMA 等 decoder 模型的 model_type 不含 "bert"
    model_type = cfg.get("model_type", "").lower()
    return "bert" not in model_type and "roberta" not in model_type


class EmbeddingEncoder:
    """
    通用向量编码器，自动适配 BERT 系（mean pooling）和 Decoder 系（last token pooling）。
    - BGE / bge-large-zh-v1.5：mean pooling
    - Qwen3-Embedding：last token pooling + 任务指令前缀
    """

    def __init__(self, model_path: str):
        print(f"加载向量模型: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        self.is_decoder = _is_decoder_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16 if self.is_decoder else torch.float32)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        arch = "Decoder（last token pooling）" if self.is_decoder else "BERT（mean pooling）"
        print(f"模型架构: {arch}，设备: {self.device}")

    def _apply_instruction(self, texts: List[str], instruction: str) -> List[str]:
        """为 Qwen3-Embedding 拼接任务指令前缀"""
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _last_token_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 找每条序列最后一个非 padding token 的位置
        last_idx = attention_mask.sum(dim=1) - 1
        batch_size = token_embeddings.size(0)
        return token_embeddings[torch.arange(batch_size, device=token_embeddings.device), last_idx]

    def encode(self, texts: List[str], normalize_embeddings: bool = True,
               show_progress_bar: bool = True, instruction: str = QWEN3_TASK_INSTRUCTION) -> np.ndarray:
        """
        将文本编码为向量。
        Decoder 模型自动拼接任务指令；BERT 模型忽略 instruction。
        """
        from tqdm import tqdm

        if self.is_decoder:
            texts = self._apply_instruction(texts, instruction)

        embeddings = []
        batch_size = 16 if self.is_decoder else 32
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

                if self.is_decoder:
                    sent_emb = self._last_token_pool(token_emb, attn_mask)
                else:
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
    Complete-linkage 聚类：只有两个簇内所有成员两两相似度都超过阈值才合并。
    比贪心聚类更保守，避免"链式错误"——即 A≈B、B≈C 但 A≉C 时三者被错误合并。
    额外返回每个合并对的相似度分数，用于日志核查。

    Returns:
        (canonical_map, merge_scores):
            canonical_map  - {原始实体: 标准名}
            merge_scores   - {原始实体: (标准名, 与标准名的相似度)}
    """
    sim = cosine_sim_matrix(embeddings)
    n = len(items)

    # 初始每个实体自成一簇，存储索引集合
    clusters: List[set] = [{i} for i in range(n)]

    # 迭代合并，直到没有可合并的簇对
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # complete-linkage：两簇所有成员对都必须超过阈值
                if all(sim[a, b] >= threshold for a in clusters[i] for b in clusters[j]):
                    clusters[i] = clusters[i] | clusters[j]
                    clusters.pop(j)
                    changed = True
                    break  # 重新从头扫描
            if changed:
                break

    # 为每个簇选标准名（最短字符串，长度相同时取字典序最小）
    canonical: Dict[str, str] = {}
    merge_scores: Dict[str, tuple] = {}

    for cluster in clusters:
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
    3. 用BGE模型将实体转换为向量
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
    from pathlib import Path    # Step 1: 加载schema中的预定义关系
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
        print(f"警告: 过滤了 {len(invalid_relations)} 个无效关系: {invalid_relations}")

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
        in_jsonl: 输入文件路径（OneKE抽取的三元组，JSONL格式）
        out_jsonl: 输出文件路径（归一化后的三元组）
        model_name: 向量模型路径，支持 Qwen3-Embedding / BGE 系列，自动适配 pooling 方式
        entity_threshold: 实体相似度阈值，使用 complete-linkage 聚类
        schema_path: schema配置文件路径
        log_path: 日志文件路径（记录归一化合并详情）
    Returns:
        归一化后的三元组数量
    """
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
        --entity-threshold 0.85
    """

    parser = argparse.ArgumentParser(description="实体归一化和关系过滤")
    parser.add_argument("--in-jsonl", required=True, help="输入JSONL文件路径（OneKE抽取结果）")
    parser.add_argument("--out-jsonl", required=True, help="输出JSONL文件路径（归一化后结果）")
    parser.add_argument("--embedding-model", default="model/Qwen3-Embedding-0.6B",
                        help="向量模型路径，支持 Qwen3-Embedding / BGE 系列，自动适配 pooling 方式")
    parser.add_argument("--schema-path", default="config/relation_types.json", help="schema配置文件路径")
    parser.add_argument("--entity-threshold", type=float, default=0.93,
                        help="实体相似度阈值（complete-linkage），越高越严格，建议 0.93~0.95")
    parser.add_argument("--log-path", default="output/logs/normalization.log",
                        help="归一化日志文件路径")
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