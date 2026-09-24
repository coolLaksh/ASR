#!/usr/bin/env bash
# One-shot environment setup: conda env, AI4Bharat NeMo fork, remaining pip deps.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="marathi_asr"
PYTHON_VERSION="3.10"
NEMO_DIR="${REPO_DIR}/NeMo"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Creating conda env ${ENV_NAME} (python ${PYTHON_VERSION})"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

pip install --upgrade pip
# reinstall.sh's `wget` dependency fails to build under newer setuptools
# (github.com/AI4Bharat/IndicConformerASR/issues/5); pin below it first.
pip install "setuptools<70"

if [ ! -d "${NEMO_DIR}" ]; then
    echo "Cloning AI4Bharat NeMo fork"
    git clone https://github.com/AI4Bharat/NeMo.git "${NEMO_DIR}"
fi
pushd "${NEMO_DIR}" > /dev/null
git checkout nemo-v2
bash reinstall.sh
popd > /dev/null

pip install -r "${REPO_DIR}/requirements.txt"

# nemo_toolkit's own dependency chain (plus requirements.txt's unpinned
# `datasets`) drifts huggingface_hub/pytorch-lightning/setuptools/torch to
# incompatible versions; pin them all together in one resolve rather than
# sequentially, so pip solves them jointly instead of each fix re-breaking
# the last.
pip install "setuptools==79.0.1" "huggingface_hub==0.23.2" "pytorch-lightning==2.2.1" \
    "torch==2.14.0" "torchvision==0.29.0" "datasets==2.19.0"

# reinstall.sh's numba pin segfaults on this machine's CUDA driver; 0.67.0 is
# the first version confirmed crash-free and numerically correct here.
# scripts/finetune.py separately works around a remaining incompatibility in
# NeMo's vendored RNNT loss kernel by using the pure-PyTorch loss backend.
pip install "numba==0.67.0"

echo "--- Verifying install ---"
python - <<'PY'
import torch
import nemo
import nemo.collections.asr as nemo_asr

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
print("nemo:", nemo.__version__)
PY
echo "--- setup.sh done ---"
