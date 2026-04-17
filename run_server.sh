#!/usr/bin/env bash
# 本脚本用于在服务器上一键执行依赖安装与OneKE+BGE全流程构建。
set -euo pipefail

# =========================
# AeroMorpho-KG server runner
# =========================
#
# 用法：
#   chmod +x run_server.sh
#   ./run_server.sh
#
# 可选环境变量（覆盖默认值）：
#   CONDA_ENV=kg_env
#   PDF_DIR=变构飞行器
#   PARSE_DIR=data/parsed
#   RAW_TRIPLES=data/triples_raw/triples_oneke.jsonl
#   FUSED_TRIPLES=data/triples_fused/triples_bge.jsonl
#   SCHEMA_PATH=config/schema.json
#   ONEKE_MODEL=zjunlp/OneKE
#   BGE_MODEL=BAAI/bge-base-zh-v1.5
#   ENTITY_THRESHOLD=0.85
#   RELATION_THRESHOLD=0.9
#   CHUNK_CHARS=1200
#   SKIP_PARSE=0              # 1 表示跳过解析层
#   DISABLE_4BIT=0            # 1 表示关闭4bit量化
#
# 必需：
#   export LLAMA_CLOUD_API_KEY=xxx

CONDA_ENV="${CONDA_ENV:-kg_env}"
PDF_DIR="${PDF_DIR:-变构飞行器}"
PARSE_DIR="${PARSE_DIR:-data/parsed}"
RAW_TRIPLES="${RAW_TRIPLES:-data/triples_raw/triples_oneke.jsonl}"
FUSED_TRIPLES="${FUSED_TRIPLES:-data/triples_fused/triples_bge.jsonl}"
SCHEMA_PATH="${SCHEMA_PATH:-config/schema.json}"
ONEKE_MODEL="${ONEKE_MODEL:-zjunlp/OneKE}"
BGE_MODEL="${BGE_MODEL:-BAAI/bge-base-zh-v1.5}"
ENTITY_THRESHOLD="${ENTITY_THRESHOLD:-0.85}"
RELATION_THRESHOLD="${RELATION_THRESHOLD:-0.9}"
CHUNK_CHARS="${CHUNK_CHARS:-1200}"
SKIP_PARSE="${SKIP_PARSE:-0}"
DISABLE_4BIT="${DISABLE_4BIT:-0}"

if [[ -z "${LLAMA_CLOUD_API_KEY:-}" && "${SKIP_PARSE}" != "1" ]]; then
  echo "[ERROR] LLAMA_CLOUD_API_KEY 未设置，且你没有跳过解析层。"
  exit 1
fi

echo "[INFO] Activating conda env: ${CONDA_ENV}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "[ERROR] conda 未安装或不在 PATH 中。"
  exit 1
fi

echo "[INFO] Installing dependencies"
pip install -r requirements.txt

PIPELINE_CMD=(
  python src/pipeline.py
  --pdf-dir "${PDF_DIR}"
  --parse-dir "${PARSE_DIR}"
  --raw-triples "${RAW_TRIPLES}"
  --fused-triples "${FUSED_TRIPLES}"
  --schema-path "${SCHEMA_PATH}"
  --oneke-model "${ONEKE_MODEL}"
  --bge-model "${BGE_MODEL}"
  --entity-threshold "${ENTITY_THRESHOLD}"
  --relation-threshold "${RELATION_THRESHOLD}"
  --chunk-chars "${CHUNK_CHARS}"
)

if [[ "${SKIP_PARSE}" == "1" ]]; then
  PIPELINE_CMD+=(--skip-parse)
fi
if [[ "${DISABLE_4BIT}" == "1" ]]; then
  PIPELINE_CMD+=(--disable-4bit)
fi

echo "[INFO] Running pipeline"
"${PIPELINE_CMD[@]}"

echo "[INFO] Generating evaluation samples"
python src/evaluate.py \
  --fused-triples "${FUSED_TRIPLES}" \
  --out-concepts "data/eval/sample_concepts.csv" \
  --out-relations "data/eval/sample_relations.csv"

echo "[DONE] All stages completed."
echo "[DONE] fused triples: ${FUSED_TRIPLES}"
echo "[DONE] eval files: data/eval/sample_concepts.csv, data/eval/sample_relations.csv"
