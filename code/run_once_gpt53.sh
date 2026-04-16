#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  OPENAI_API_KEY="$(
    python - <<'PY'
import json
import os
from pathlib import Path

paths = []
custom_auth = os.getenv("CODEX_AUTH_FILE", "").strip()
if custom_auth:
    paths.append(Path(custom_auth).expanduser())
paths.extend([
    Path("~/.codex/auth.json").expanduser(),
    Path("~/.config/codex/auth.json").expanduser(),
])

for p in paths:
    if not p.is_file():
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        key = data.get("OPENAI_API_KEY") or data.get("openai_api_key")
        if key:
            print(key)
            raise SystemExit(0)
    except Exception:
        continue
PY
  )"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set and was not found in auth.json."
  echo "Run: export OPENAI_API_KEY='your_api_key'  (or configure ~/.codex/auth.json)"
  exit 1
fi

API_URL="${API_URL:-https://api2.tabcode.cc/openai}"
TOPIC="${TOPIC:-$(head -n 1 topics_demo.txt)}"
MODEL="${MODEL:-gpt-5-codex}"
SAVE_PATH="${SAVE_PATH:-./output/res/gpt5_codex_once}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WIRE_API="${WIRE_API:-responses}"
DEBUG="${DEBUG:-0}"

echo "Topic: $TOPIC"
echo "Model: $MODEL"
echo "Save path: $SAVE_PATH"
echo "CUDA device: $CUDA_DEVICE"
echo "API URL: $API_URL"
echo "Wire API: $WIRE_API"
echo "Debug: $DEBUG"

EXTRA_ARGS=()
if [[ "$DEBUG" == "1" ]]; then
  EXTRA_ARGS+=(--debug)
fi

WIRE_API="$WIRE_API" CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python main.py \
  --topic "$TOPIC" \
  --model "$MODEL" \
  --api_key "$OPENAI_API_KEY" \
  --api_url "$API_URL" \
  --db_path "./sf_assets/database" \
  --embedding_model "./sf_assets/gte-large-en-v1.5" \
  --survey_outline_path "./sf_assets" \
  --saving_path "$SAVE_PATH" \
  "${EXTRA_ARGS[@]}"

echo "Done. Outputs under: $SAVE_PATH"
