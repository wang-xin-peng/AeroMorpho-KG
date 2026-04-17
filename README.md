## AeroMorpho-KG

变构飞行器知识图谱构建项目（课程大作业）。

### 目标

- 从 `变构飞行器` 目录中的 PDF 文献抽取知识图谱。
- 满足作业要求：概念不少于 500，关系不少于 1000。
- 生成评估抽样：200 个概念、400 条关系用于人工标注。
- 支持 Neo4j 入库与查询展示。

### 目录结构

- `变构飞行器/`：原始论文数据集
- `data/parsed/`：LlamaParse 解析后的 Markdown
- `data/triples_raw/`：抽取得到的原始三元组
- `data/triples_fused/`：融合归一化后的三元组
- `data/eval/`：人工评估抽样结果
- `src/`：各层流水线代码

### 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

1. 配置环境变量（复制 `.env.example` 为 `.env` 并填写）

```bash
copy .env.example .env
```

1. 执行完整流水线（DeepKE + OneKE + BGE）

```bash
python src/pipeline.py ^
  --pdf-dir "变构飞行器" ^
  --parse-dir "data/parsed" ^
  --raw-triples "data/triples_raw/triples_oneke.jsonl" ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --schema-path "config/schema.json" ^
  --oneke-model "zjunlp/OneKE" ^
  --bge-model "BAAI/bge-base-zh-v1.5"
```

1. 生成人工评估抽样（200 概念 + 400 关系）

```bash
python src/evaluate.py ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --out-concepts "data/eval/sample_concepts.csv" ^
  --out-relations "data/eval/sample_relations.csv"
```

1. 导入 Neo4j

```bash
python src/load_to_neo4j.py --triples "data/triples_fused/triples_bge.jsonl"
```

### 技术路线（按课程方案）

- Layer 1：LlamaCloud 文档解析（`src/parse_docs.py`）
- Layer 2：DeepKE + OneKE Schema 抽取（`src/extract_triples_oneke.py`）
- Layer 3：BGE 向量融合归一化（`src/fuse_entities_bge.py`）
- Layer 4：Neo4j 存储与查询（`src/load_to_neo4j.py`）
- Layer 5：评估抽样与应用展示（`src/evaluate.py`、`src/app_demo.py`）

### OneKE + BGE 执行命令（同上）

```bash
python src/pipeline.py ^
  --pdf-dir "变构飞行器" ^
  --parse-dir "data/parsed" ^
  --raw-triples "data/triples_raw/triples_oneke.jsonl" ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --schema-path "config/schema.json" ^
  --oneke-model "zjunlp/OneKE" ^
  --bge-model "BAAI/bge-base-zh-v1.5"
```

