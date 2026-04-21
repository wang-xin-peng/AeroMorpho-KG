# 知识图谱构建流水线

## 流程概览

```
PDF文档
  ↓
[1. 解析] → parse_docs.py
  ↓
Markdown
  ↓
[2. 预处理] → preprocess_docs.py
  ↓
清理后的Markdown（去除元数据、参考文献等）
  ↓
[3. 抽取] → extract_triples.py (OneKE LLM)
  ↓
原始三元组（带噪声、冗余）
  ↓
[4. 归一化和过滤] → normalize_and_filter.py
  • 实体归一化（基于语义相似度聚类）
  • 关系过滤（验证是否在预定义schema中）
  ↓
归一化后的三元组（实体已对齐）
  ↓
[5. 后处理] → postprocess.py
  • 实体类型标注
  • 类型约束检查
  • 对称关系补全
  • 互斥关系冲突解决
  ↓
高质量三元组
```

## 各阶段详细说明

### 1. 解析阶段 (parse_docs.py)
- **输入**: `data/raw/*.pdf`
- **输出**: `data/parsed/*.md`
- **功能**: 使用LlamaParse将PDF转换为Markdown格式

### 2. 预处理阶段 (preprocess_docs.py)
- **输入**: `data/parsed/*.md`
- **输出**: `data/preprocessed/*.md`
- **功能**:
  - 移除作者信息、机构、邮箱
  - 移除页码、页眉、页脚
  - 移除URL和DOI
  - 移除引用标记 [1], [1-3]
  - 移除论文元数据（目录、图表列表、符号表等）
  - 智能移除参考文献（区分书籍和论文）
  - 转换HTML表格为Markdown
  - 保留图片描述，移除图片标签

### 3. 抽取阶段 (extract_triples.py)
- **输入**: `data/preprocessed/*.md`
- **输出**: `data/triples_raw/triples.jsonl`
- **功能**:
  - 使用OneKE大模型抽取知识三元组
  - 句子边界分块 + 滑动窗口重叠策略
  - 参数配置:
    - `chunk_chars=150` (极小chunk，最大化召回)
    - `overlap=30` (20%重叠)
    - `max_new_tokens=300` (官方推荐)
    - `split_num=4` (RE任务官方推荐)
    - `load_in_4bit=True` (默认开启4bit量化)

### 4. 归一化和过滤阶段 (normalize_and_filter.py)
- **输入**: `data/triples_raw/triples.jsonl`
- **输出**: `data/triples_normalized/triples.jsonl`
- **功能**:
  - **实体归一化**: 使用Yuan-Embedding计算实体语义向量，基于余弦相似度聚类合并同义实体
  - **关系过滤**: 验证关系是否在`config/relation_types.json`预定义的36个关系中
  - **去重**: 移除重复的三元组
- **参数**:
  - `entity_threshold=0.85` (实体相似度阈值)

### 5. 后处理阶段 (postprocess.py)
- **输入**: `data/triples_normalized/triples.jsonl`
- **输出**: `data/triples_final/triples.jsonl`
- **功能**:

#### 5.1 实体类型标注
- 为每个实体分配类型标签（12个预定义类型）
- 策略：
  1. 构建类型原型向量（类型名 + 描述 + 示例实体）
  2. 计算实体向量与类型原型向量的余弦相似度
  3. 取最高分且超过阈值的类型作为标签
- 参数: `type_threshold=0.6`

#### 5.2 类型约束检查
- 验证三元组的头尾实体类型是否符合关系定义
- 依赖: `relation_types.json`中的`head_types`和`tail_types`字段（待添加）
- 不符合约束的三元组将被过滤

#### 5.3 对称关系补全
- 对于标记为对称的关系，自动生成反向三元组
- 依赖: `relation_types.json`中的`is_symmetric`字段（待添加）
- 例如: 如果"A-对比-B"存在，自动补全"B-对比-A"

#### 5.4 互斥关系冲突解决
- 检测并解决互斥关系之间的冲突
- 依赖: `relation_types.json`中的`mutex_with`字段（待添加）
- 策略: 保留置信度更高或来源更可靠的三元组

## 配置文件

### config/relation_types.json
当前包含36个预定义关系，每个关系包含：
- `name`: 关系名称
- `description`: 关系描述

**待添加字段**:
- `head_types`: 头实体允许的类型集合（用于类型约束检查）
- `tail_types`: 尾实体允许的类型集合（用于类型约束检查）
- `is_symmetric`: 是否为对称关系（用于对称补全）
- `mutex_with`: 互斥关系列表（用于冲突解决）

示例：
```json
{
  "name": "具有",
  "description": "实体具备某种属性、能力或特征",
  "head_types": ["变形方式", "气动布局"],
  "tail_types": ["飞行性能与稳定性"],
  "is_symmetric": false,
  "mutex_with": []
}
```

### config/entity_types.json
包含12个预定义实体类型：
1. 变形方式
2. 变形尺度
3. 驱动方式
4. 控制策略
5. 结构材料
6. 感知与传感
7. 气动布局
8. 动力布局
9. 任务应用
10. 飞行性能与稳定性
11. 工程成本与效益
12. 变体结构优化方法

每个类型包含：
- `name`: 类型名称
- `description`: 类型描述
- `examples`: 代表性实体示例

## 运行方式

### 完整流水线
```bash
python src/pipeline.py
```

### 单独运行各阶段

#### 1. 解析
```bash
python src/parse_docs.py --pdf-dir ./data/raw --out-dir ./data/parsed
```

#### 2. 预处理
```bash
python src/preprocess_docs.py --input-dir ./data/parsed --output-dir ./data/preprocessed
```

#### 3. 抽取
```bash
python src/extract_triples.py \
  --parsed-dir ./data/preprocessed \
  --out-jsonl ./output/triples_raw/triples.jsonl \
  --schema-path ./config/relation_types.json
```

#### 4. 归一化和过滤
```bash
python src/normalize_and_filter.py \
  --in-jsonl ./output/triples_raw/triples.jsonl \
  --out-jsonl ./output/triples_normalized/triples.jsonl \
  --entity-threshold 0.85
```

#### 5. 后处理
```bash
python src/postprocess.py \
  --in-jsonl ./output/triples_normalized/triples.jsonl \
  --out-jsonl ./output/triples_final/triples.jsonl \
  --type-threshold 0.6
```

## 目录结构

```
AeroMorpho-KG/
├── config/
│   ├── entity_types.json       # 12个实体类型定义
│   └── relation_types.json     # 36个关系类型定义
├── data/
│   ├── raw/                    # 原始PDF文档
│   ├── parsed/                 # 解析后的Markdown
│   └── preprocessed/           # 预处理后的Markdown
├── output/
│   ├── triples_raw/            # 原始抽取的三元组
│   ├── triples_normalized/     # 归一化后的三元组
│   └── triples_final/          # 最终高质量三元组
├── src/
│   ├── parse_docs.py           # [1] PDF解析
│   ├── preprocess_docs.py      # [2] 文档预处理
│   ├── extract_triples.py      # [3] 知识抽取
│   ├── normalize_and_filter.py # [4] 归一化和过滤
│   ├── postprocess.py          # [5] 后处理
│   ├── pipeline.py             # 完整流水线
│   ├── schema.py               # Schema工具类
│   └── common.py               # 通用工具函数
└── model/
    ├── OneKE/                  # OneKE抽取模型
    └── Yuan-Embedding/         # Yuan-Embedding向量模型
```

## 参数调优建议

### 抽取阶段
- `chunk_chars`: 150-300，越小召回率越高但速度越慢
- `overlap`: chunk_chars的20%，避免边界信息丢失
- `entity_threshold`: 0.8-0.9，越高实体归一化越严格

### 后处理阶段
- `type_threshold`: 0.5-0.7，越高类型标注越严格，未标注实体越多

## 下一步工作

1. **完善relation_types.json**:
   - 为每个关系添加`head_types`和`tail_types`
   - 标记对称关系（`is_symmetric`）
   - 定义互斥关系（`mutex_with`）

2. **测试后处理功能**:
   - 验证类型约束检查是否正确
   - 验证对称补全是否合理
   - 验证互斥冲突解决策略

3. **评估质量**:
   - 统计各阶段的数据量变化
   - 人工抽样评估准确率
   - 调整阈值参数优化效果
