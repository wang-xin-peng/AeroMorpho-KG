## AeroMorpho-KG

变构飞行器知识图谱构建项目（课程大作业）。

### 目标

- 从 `data/raw` 目录中的 PDF 文献抽取知识图谱。
- 满足作业要求：概念不少于 500，关系不少于 1000。
- 生成评估抽样：200 个概念、400 条关系用于人工标注。
- 支持 Neo4j 入库与查询展示。

### 目录结构

- `data/raw/`：原始论文PDF数据集
- `data/parsed/`：LlamaParse 解析后的 Markdown
- `data/triples_raw/`：抽取得到的原始三元组
- `data/triples_fused/`：融合归一化后的三元组
- `data/eval/`：人工评估抽样结果
- `src/`：各层流水线代码

### 快速开始

下载模型

pip install modelscope

modelscope download --model ZJUNLP/OneKE --local_dir ./model/OneKE

modelscope download --model IEITYuan/Yuan-embedding-2.0-zh --local_dir ./model/Yuan-Embedding

1. 安装依赖

```bash
pip install -r requirements.txt
```

1. 配置环境变量（复制 `.env.example` 为 `.env` 并填写）

```bash
copy .env.example .env
```

1. 执行完整流水线（DeepKE + OneKE + Yuan-Embedding）

```bash
python src/pipeline.py ^
  --pdf-dir "data/raw" ^
  --parse-dir "data/parsed" ^
  --raw-triples "data/triples_raw/triples_oneke.jsonl" ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --schema-path "config/schema.json" ^
  --oneke-model "model/OneKE" ^
  --embedding-model "model/Yuan-Embedding"
```

1. 生成人工评估抽样（200 概念 + 400 关系）

```bash
python src/evaluate.py ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --out-concepts "data/eval/sample_concepts.csv" ^
  --out-relations "data/eval/sample_relations.csv"
```

1. 运行知识图谱评估（准确性、一致性、完整性）

**快速启动（推荐）：**
```bash
# Windows
run_evaluation.bat

# Linux/Mac
bash run_evaluation.sh
```

**或手动运行：**
```bash
# 检查评估准备状态
python src/eval_helper.py --check

# 生成标注指南
python src/eval_helper.py --guide

# 完成人工标注后，运行综合评估
python src/eval_all.py ^
  --triples data/triples_fused/triples_fused.jsonl ^
  --schema config/schema.json ^
  --entity-csv data/eval/sample_concepts.csv ^
  --relation-csv data/eval/sample_relations.csv ^
  --output-dir data/eval
```

详细说明请参考：[评估系统使用说明.md](评估系统使用说明.md)

1. 导入 Neo4j

```bash
python src/load_to_neo4j.py --triples "data/triples_fused/triples_bge.jsonl"
```

### 技术路线（按课程方案）

- Layer 1：LlamaCloud 文档解析（`src/parse_docs.py`）
- Layer 2：DeepKE + OneKE Schema 抽取（`src/extract_triples_oneke.py`）
- Layer 3：Yuan-Embedding 向量融合归一化（`src/fuse_entities_bge.py`）
- Layer 4：Neo4j 存储与查询（`src/load_to_neo4j.py`）
- Layer 5：评估抽样与应用展示（`src/evaluate.py`、`src/app_demo.py`）

### 评估系统

本项目实现了完整的知识图谱评估系统，包含三个维度：

#### 评估维度

1. **准确性评估** (`src/eval_accuracy.py`)
   - 计算准确率、召回率、F1分数
   - 分析错误类型分布
   - 支持实体和关系的独立评估

2. **一致性评估** (`src/eval_consistency.py`)
   - 检测同义实体
   - 检测关系对称性违规
   - 检测关系传递性缺失
   - 检测关系互斥冲突
   - 检测Schema违规

3. **完整性评估** (`src/eval_completeness.py`)
   - 评估实体覆盖度
   - 评估关系覆盖度
   - 计算实体连接度
   - 计算知识图谱密度

#### 评估工具

- `src/evaluate.py`：生成标注模板（200概念+400关系）
- `src/eval_helper.py`：检查评估状态、生成标注指南
- `src/eval_all.py`：综合评估（整合三个维度）
- `test_evaluation_system.py`：自动化测试脚本
- `run_evaluation.bat/sh`：快速启动脚本

#### 评估流程

```
生成标注模板 → 人工标注 → 运行评估 → 查看报告
```

详细文档：
- [评估系统使用说明.md](评估系统使用说明.md)
- [评估系统实现总结.md](评估系统实现总结.md)
- [data/eval/annotation_guide.txt](data/eval/annotation_guide.txt)

### OneKE + Yuan-Embedding 执行命令（同上）

```bash
python src/pipeline.py ^
  --pdf-dir "data/raw" ^
  --parse-dir "data/parsed" ^
  --raw-triples "data/triples_raw/triples_oneke.jsonl" ^
  --fused-triples "data/triples_fused/triples_bge.jsonl" ^
  --schema-path "config/schema.json" ^
  --oneke-model "model/OneKE" ^
  --embedding-model "model/Yuan-Embedding"
```

