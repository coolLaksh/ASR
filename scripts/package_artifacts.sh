#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "${BASH_SOURCE[0]}") <run_id>" >&2
  exit 1
fi

RUN_ID="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/${RUN_ID}"

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "Error: artifacts directory not found: ${ARTIFACT_DIR}" >&2
  exit 1
fi

EXPECTED_FILES=(checkpoint.nemo train.log config_used.yaml eval_wer.json)
MISSING_FILES=()
for f in "${EXPECTED_FILES[@]}"; do
  if [[ ! -f "${ARTIFACT_DIR}/${f}" ]]; then
    MISSING_FILES+=("${f}")
  fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
  echo "Warning: missing expected files in ${ARTIFACT_DIR}:" >&2
  for f in "${MISSING_FILES[@]}"; do
    echo "  - ${f}" >&2
  done
else
  echo "All expected files present in ${ARTIFACT_DIR}."
fi

TARBALL="${REPO_ROOT}/artifacts/${RUN_ID}.tar.gz"
tar -czf "${TARBALL}" -C "${REPO_ROOT}/artifacts" "${RUN_ID}"

echo "Packaged artifacts: ${TARBALL}"
du -h "${TARBALL}"
