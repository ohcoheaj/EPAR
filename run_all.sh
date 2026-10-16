#!/bin/bash
# Supply cluster-specific PBS options with qsub; see README.md.

set -e
set -o pipefail

EPAR_ROOT="${EPAR_ROOT:-/home/ijaehojo/EPAR}"
cd "$EPAR_ROOT"
EPAR_ROOT="$PWD"

if [[ ! -f "$EPAR_ROOT/vendor/knn_cuda/setup.py" || ! -f "$EPAR_ROOT/run_pipeline.py" ]]; then
    echo "Invalid EPAR_ROOT: $EPAR_ROOT (KNN_CUDA source or runner missing)" >&2
    exit 1
fi

source activate mmad

python -m pip uninstall -y knn_cuda
cd "$EPAR_ROOT/vendor/knn_cuda"
export TORCH_CUDA_ARCH_LIST="8.6"
python -m pip install -e . --no-build-isolation
cd "$EPAR_ROOT"

DATASET_PATH="${DATASET_PATH:-/home/ijaehojo/AD-JEPA/datasets}"
python -u "$EPAR_ROOT/run_pipeline.py" --dataset_path "$DATASET_PATH"
