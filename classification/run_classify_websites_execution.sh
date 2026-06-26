#!/usr/bin/env bash
#SBATCH --job-name=websites_classify
#SBATCH --partition=pg2tfg12
#SBATCH --qos=q_pg2tfg12
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/home/aaasifar/spai-hf/classification_results/slurm_logs/%x_%j.out
#SBATCH --error=/home/aaasifar/spai-hf/classification_results/slurm_logs/%x_%j.err

set -Eeuo pipefail

trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# =========================
# Project paths
# =========================

PROJECT_DIR="/home/aaasifar/spai-hf"
cd "$PROJECT_DIR"

WORK_DIR="$PROJECT_DIR"

# Allow Python to import inference.py from project root
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# Avoid conflicts with user-level Python packages
export PYTHONNOUSERSITE=1
unset PYTHONHOME

# Disable Albumentations update warning
export NO_ALBUMENTATIONS_UPDATE=1

# CPU thread settings
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# =========================
# Load conda
# =========================

if [[ -f /home/aaasifar/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/aaasifar/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/miniconda3/etc/profile.d/conda.sh ]]; then
  source /opt/miniconda3/etc/profile.d/conda.sh
else
  echo "[ERROR] Could not find conda.sh" >&2
  exit 1
fi

conda activate spai-hf-2

PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"

# =========================
# Default configuration
# =========================

METADATA_CSV="${METADATA_CSV:-$WORK_DIR/metadata.csv}"
IMAGES_DIR="${IMAGES_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/classification_results}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR}"
THRESHOLD="${THRESHOLD:-0.35}"
MAX_IMAGES="${MAX_IMAGES:-0}"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/slurm_logs"

# =========================
# Info
# =========================

echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] PROJECT_DIR: $PROJECT_DIR"
echo "[INFO] WORK_DIR: $WORK_DIR"
echo "[INFO] CURRENT_DIR: $(pwd)"
echo "[INFO] CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "[INFO] PYTHON_BIN: $PYTHON_BIN"
echo "[INFO] METADATA_CSV: $METADATA_CSV"
echo "[INFO] IMAGES_DIR: ${IMAGES_DIR:-<not set>}"
echo "[INFO] OUTPUT_DIR: $OUTPUT_DIR"
echo "[INFO] MODEL_DIR: $MODEL_DIR"
echo "[INFO] THRESHOLD: $THRESHOLD"
echo "[INFO] MAX_IMAGES: $MAX_IMAGES"
echo "[INFO] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] nvidia-smi:"
  nvidia-smi
else
  echo "[WARNING] nvidia-smi not found"
fi

# =========================
# Check files
# =========================

if [[ ! -f "$PROJECT_DIR/tools/classify_websites_execution.py" ]]; then
  echo "[ERROR] Python script not found: $PROJECT_DIR/tools/classify_websites_execution.py" >&2
  exit 1
fi

if [[ -n "$IMAGES_DIR" ]]; then
  if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "[ERROR] Images directory not found: $IMAGES_DIR" >&2
    exit 1
  fi
else
  if [[ ! -f "$METADATA_CSV" ]]; then
    echo "[ERROR] Metadata CSV not found: $METADATA_CSV" >&2
    exit 1
  fi
fi

# =========================
# Preflight check
# =========================

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("[INFO] Python executable:", sys.executable)
print("[INFO] torch version:", torch.__version__)
print("[INFO] CUDA available:", torch.cuda.is_available())
print("[INFO] CUDA version:", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This job requested a GPU, but PyTorch cannot see it."
    )

print("[INFO] GPU:", torch.cuda.get_device_name(0))
print("[INFO] Compute capability:", torch.cuda.get_device_capability(0))
print("[INFO] Torch supported archs:", torch.cuda.get_arch_list())

major, minor = torch.cuda.get_device_capability(0)

if major < 7:
    raise RuntimeError(
        f"The assigned GPU has compute capability {(major, minor)}, "
        "which is too old for this PyTorch build. "
        "Use RTX 2080 Ti or L40S instead of Titan."
    )

print("[INFO] GPU check passed.")
PY

# =========================
# Build command
# =========================

CMD=(
  "$PYTHON_BIN"
  -u
  "$PROJECT_DIR/tools/classify_websites_execution.py"
  --output-dir "$OUTPUT_DIR"
  --model-dir "$MODEL_DIR"
  --threshold "$THRESHOLD"
)

if [[ -n "$IMAGES_DIR" ]]; then
  CMD+=(--images-dir "$IMAGES_DIR")
else
  CMD+=(--metadata-csv "$METADATA_CSV")
fi

if [[ "$MAX_IMAGES" != "0" ]]; then
  CMD+=(--max-images "$MAX_IMAGES")
fi

# Forward extra arguments to Python script
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] Prepared command:"
printf ' %q' "${CMD[@]}"
echo

# =========================
# Run classifier
# =========================

"${CMD[@]}"

echo "[INFO] Classification completed successfully."