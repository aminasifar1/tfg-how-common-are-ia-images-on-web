#!/usr/bin/env bash
#SBATCH --job-name=single_scrape
#SBATCH --partition=pg2tfg12
#SBATCH --qos=q_pg2tfg12
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=/home/aaasifar/spai-hf/scraping_logs/%x_%j.out
#SBATCH --error=/home/aaasifar/spai-hf/scraping_logs/%x_%j.err

set -Eeuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO: $BASH_COMMAND" >&2' ERR

# =========================
# Project paths
# =========================

PROJECT_DIR="/home/aaasifar/spai-hf"
TFG_DIR="$PROJECT_DIR/tfg-how-common-are-ia-images-on-web"
SCRAPING_DIR="$TFG_DIR/scraping"

export PYTHONPATH="$SCRAPING_DIR:$TFG_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME 2>/dev/null || true

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
# Configuration
# =========================

SCRAPE_URL="${SCRAPE_URL:?Set SCRAPE_URL to the website to scrape}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/single_scrape_output}"
MAX_IMAGES="${MAX_IMAGES:-200}"
MAX_PAGES="${MAX_PAGES:-5}"
DELAY="${DELAY:-1.0}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/scraping_logs"

echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] SCRAPE_URL: $SCRAPE_URL"
echo "[INFO] OUTPUT_DIR: $OUTPUT_DIR"
echo "[INFO] MAX_IMAGES: $MAX_IMAGES"
echo "[INFO] MAX_PAGES: $MAX_PAGES"
echo "[INFO] DELAY: $DELAY"
echo "[INFO] Extra args: $*"

# =========================
# Run
# =========================

CMD=(
  "$PYTHON_BIN" -u
  "$SCRAPING_DIR/advanced_image_scraper.py"
  --url "$SCRAPE_URL"
  --output-dir "$OUTPUT_DIR"
  --max-images "$MAX_IMAGES"
  --max-pages "$MAX_PAGES"
  --delay "$DELAY"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[INFO] Single-site scraping completed."
