#!/usr/bin/env bash
# Cheap pipeline sanity check before committing to a real training run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_LIMIT=40  # not fewer: the 90/5/5 split needs this to yield a non-empty dev split
RUN_ID="smoke_test"

echo "=== [1/2] Building smoke-test manifest (--limit ${SMOKE_LIMIT}) ==="
python data/prepare_manifest.py --limit "${SMOKE_LIMIT}" --output-dir data/processed_smoke

echo "=== [2/2] Running finetune.py --max-steps 2 ==="
python scripts/finetune.py \
  --train-manifest data/processed_smoke/train_manifest.jsonl \
  --val-manifest data/processed_smoke/dev_manifest.jsonl \
  --run-id "${RUN_ID}" \
  --max-steps 2 \
  --batch-size 2

echo "=== Verifying artifacts ==="
test -f "artifacts/${RUN_ID}/checkpoint.nemo" || { echo "FAIL: checkpoint.nemo missing"; exit 1; }
test -f "artifacts/${RUN_ID}/config_used.yaml" || { echo "FAIL: config_used.yaml missing"; exit 1; }

echo "Smoke test PASSED"
