#!/usr/bin/env bash
#SBATCH --job-name=wayback_scrape
#SBATCH --partition=pg2tfg12
#SBATCH --qos=q_pg2tfg12
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
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

CSV_PATH="${CSV_PATH:-$TFG_DIR/data/websites-news-arts.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/wayback_images_by_year/output}"
START_YEAR="${START_YEAR:-2020}"
END_YEAR="${END_YEAR:-2025}"
MAX_SNAPSHOTS="${MAX_SNAPSHOTS:-5}"
MAX_IMAGES_PER_YEAR="${MAX_IMAGES_PER_YEAR:-50}"
DELAY="${DELAY:-1.5}"
CRAWL_SITE="${CRAWL_SITE:-}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/scraping_logs"

echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] CSV_PATH: $CSV_PATH"
echo "[INFO] OUTPUT_DIR: $OUTPUT_DIR"
echo "[INFO] START_YEAR: $START_YEAR"
echo "[INFO] END_YEAR: $END_YEAR"
echo "[INFO] MAX_SNAPSHOTS: $MAX_SNAPSHOTS"
echo "[INFO] MAX_IMAGES_PER_YEAR: $MAX_IMAGES_PER_YEAR"
echo "[INFO] DELAY: $DELAY"
echo "[INFO] CRAWL_SITE: ${CRAWL_SITE:-<all>}"

# =========================
# Check files
# =========================

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERROR] CSV file not found: $CSV_PATH" >&2
  exit 1
fi

# =========================
# Build command
# =========================

CMD=(
  "$PYTHON_BIN" -u
  "$SCRAPING_DIR/run_wayback_full_scrape.py"
  --csv "$CSV_PATH"
  --output "$OUTPUT_DIR"
  --start-year "$START_YEAR"
  --end-year "$END_YEAR"
  --max-snapshots-per-year "$MAX_SNAPSHOTS"
  --max-images-per-year "$MAX_IMAGES_PER_YEAR"
  --delay "$DELAY"
)

if [[ -n "$CRAWL_SITE" ]]; then
  CMD+=(--site-index "$CRAWL_SITE")
fi

echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[INFO] Wayback scraping completed successfully."
