#!/usr/bin/env bash
#SBATCH --job-name=web_scraper
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
# Configuration (env vars with defaults)
# =========================

CSV_PATH="${CSV_PATH:-$TFG_DIR/data/websites-list.csv}"
BASE_OUTPUT="${BASE_OUTPUT:-$PROJECT_DIR/batch_scrape_results}"
MAX_PAGES="${MAX_PAGES:-5}"
MIN_IMAGES_PER_PAGE="${MIN_IMAGES_PER_PAGE:-0}"
MAX_IMAGES_PER_SITE="${MAX_IMAGES_PER_SITE:-200}"
DELAY="${DELAY:-1.0}"
CRAWL_SITE="${CRAWL_SITE:-}"

mkdir -p "$BASE_OUTPUT"
mkdir -p "$PROJECT_DIR/scraping_logs"

# =========================
# Info
# =========================

echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] CSV_PATH: $CSV_PATH"
echo "[INFO] BASE_OUTPUT: $BASE_OUTPUT"
echo "[INFO] MAX_PAGES: $MAX_PAGES"
echo "[INFO] MIN_IMAGES_PER_PAGE: $MIN_IMAGES_PER_PAGE"
echo "[INFO] MAX_IMAGES_PER_SITE: $MAX_IMAGES_PER_SITE"
echo "[INFO] DELAY: $DELAY"
echo "[INFO] CRAWL_SITE: ${CRAWL_SITE:-<all>}"
echo "[INFO] Extra args: $*"

# =========================
# Check files
# =========================

if [[ ! -f "$CSV_PATH" ]]; then
  echo "[ERROR] CSV file not found: $CSV_PATH" >&2
  exit 1
fi

if [[ ! -f "$SCRAPING_DIR/batch_scraper.py" ]]; then
  echo "[ERROR] batch_scraper.py not found in: $SCRAPING_DIR" >&2
  exit 1
fi

# =========================
# Build command
# =========================

CMD=(
  "$PYTHON_BIN" -u
  "$SCRAPING_DIR/batch_scraper.py"
  --csv "$CSV_PATH"
  --output-dir "$BASE_OUTPUT"
  --max-images "$MAX_IMAGES_PER_SITE"
  --max-pages "$MAX_PAGES"
  --min-images-per-page "$MIN_IMAGES_PER_PAGE"
  --delay "$DELAY"
)

if [[ -n "$CRAWL_SITE" ]]; then
  CMD+=(--site-index "$CRAWL_SITE")
fi

# Forward extra CLI arguments (e.g. --use-playwright-fallback --accept-cookies)
if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "[INFO] Command:"
printf ' %q' "${CMD[@]}"
echo

# =========================
# Run scraper
# =========================

"${CMD[@]}"

echo "[INFO] Scraping completed successfully."
