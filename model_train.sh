#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$ROOT_DIR/data/open_data/criteo_sample/normalized}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$ROOT_DIR/data/open_data/criteo_sample/normalized}"
PREDICT_DATA_DIR="${PREDICT_DATA_DIR:-$ROOT_DIR/data/open_data/criteo_sample/normalized}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model_dir}"
EXPORT_DIR="${EXPORT_DIR:-$ROOT_DIR/export_dir}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$MODEL_DIR" "$EXPORT_DIR"

"$PYTHON_BIN" "$ROOT_DIR/model_train.py" \
  --batch_size=128 \
  --num_epochs=1 \
  --run_mode="train" \
  --train_data_dir="$TRAIN_DATA_DIR" \
  --eval_data_dir="$EVAL_DATA_DIR" \
  --predict_data_dir="$PREDICT_DATA_DIR" \
  --model_dir="$MODEL_DIR" \
  --export_dir="$EXPORT_DIR" \
  --export_signature_def="banner" \
  --deep_lr=0.005 \
  --deep_l1=0.001 \
  --deep_l2=0.01 \
  --wide_lr=0.005 \
  --wide_l1=0.3 \
  --wide_l2=0.01
