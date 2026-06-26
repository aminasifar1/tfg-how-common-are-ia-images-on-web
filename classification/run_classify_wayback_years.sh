#!/usr/bin/env bash
#SBATCH --job-name=wayback_classify
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

SITES_DIR="${SITES_DIR:-$WORK_DIR/wayback_images_by_year/output/sites}"
SITE="${SITE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/wayback_classification_results}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR}"
THRESHOLD="${THRESHOLD:-0.35}"

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
echo "[INFO] SITES_DIR: $SITES_DIR"
echo "[INFO] SITE: ${SITE:-<all>}"
echo "[INFO] OUTPUT_DIR: $OUTPUT_DIR"
echo "[INFO] MODEL_DIR: $MODEL_DIR"
echo "[INFO] THRESHOLD: $THRESHOLD"
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

if [[ ! -f "$PROJECT_DIR/tools/classify_wayback_years.py" ]]; then
  echo "[ERROR] Python script not found: $PROJECT_DIR/tools/classify_wayback_years.py" >&2
  exit 1
fi

if [[ ! -d "$SITES_DIR" ]]; then
  echo "[ERROR] Sites directory not found: $SITES_DIR" >&2
  exit 1
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
# Collect site folders
# =========================

SITE_DIRS=()
if [[ -n "$SITE" ]]; then
  if [[ ! -d "$SITES_DIR/$SITE" ]]; then
    echo "[ERROR] Site folder not found: $SITES_DIR/$SITE" >&2
    exit 1
  fi
  SITE_DIRS+=("$SITES_DIR/$SITE")
else
  for d in "$SITES_DIR"/*/; do
    [[ -d "$d" ]] || continue
    SITE_DIRS+=("${d%/}")
  done
fi

if [[ "${#SITE_DIRS[@]}" -eq 0 ]]; then
  echo "[ERROR] No site folders found under: $SITES_DIR" >&2
  exit 1
fi

echo "[INFO] Sites to process (${#SITE_DIRS[@]}):"
for d in "${SITE_DIRS[@]}"; do
  echo "  - $(basename "$d")"
done

# =========================
# Run classifier per site (one --root-dir per site, with year subfolders)
# =========================

FAILED_SITES=()

for site_dir in "${SITE_DIRS[@]}"; do
  site_name="$(basename "$site_dir")"

  has_year_dir=false
  for yd in "$site_dir"/*/; do
    [[ -d "$yd" ]] || continue
    yname="$(basename "${yd%/}")"
    if [[ "$yname" =~ ^[0-9]+$ ]]; then
      has_year_dir=true
      break
    fi
  done

  if [[ "$has_year_dir" == false ]]; then
    echo "[SKIP] $site_name: no se encontraron subcarpetas de año"
    continue
  fi

  echo "[INFO] === Clasificando sitio: $site_name ==="

  CMD=(
    "$PYTHON_BIN"
    -u
    "$PROJECT_DIR/tools/classify_wayback_years.py"
    --root-dir "$site_dir"
    --output-dir "$OUTPUT_DIR/$site_name"
    --model-dir "$MODEL_DIR"
    --threshold "$THRESHOLD"
  )

  echo "[INFO] Comando:"
  printf ' %q' "${CMD[@]}"
  echo

  if ! "${CMD[@]}"; then
    echo "[WARNING] Fallo al clasificar $site_name, continuando con el siguiente sitio" >&2
    FAILED_SITES+=("$site_name")
  fi
done

if [[ "${#FAILED_SITES[@]}" -gt 0 ]]; then
  echo "[WARNING] Sitios con errores (${#FAILED_SITES[@]}):"
  for s in "${FAILED_SITES[@]}"; do
    echo "  - $s"
  done
fi

echo "[INFO] Clasificación de wayback completada."
