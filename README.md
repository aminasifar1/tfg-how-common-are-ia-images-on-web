# How Common Are AI-Generated Images on the Web?
---

## Main Context

This project studies the prevalence of AI-generated images across publicly accessible websites. It combines an automated web scraper, a Wayback Machine historical scraper, and the SPAI spectral classifier to collect and classify images from 25 websites across five sectors: news, arts and illustration, education, e-commerce, and tourism.

The [SPAI model](https://github.com/mever-team/spai) (Karageorgiou et al., CVPR 2025) is a spectral learning classifier that assigns each image a score between 0 (real) and 1 (AI-generated), using a threshold of **0.35**.

## Datasets

The collected image datasets are hosted on [Hugging Face](https://huggingface.co/datasets/aminasifar1/tfg-web-images):

| File | Description |
|------|-------------|
| `scrape_results_websites.tar.gz` | Images scraped from 25 websites across 5 sectors |
| `wayback_results_news.tar.gz` | Historical images from 5 news websites via Wayback Machine (2020-2025) |

---

## Project Structure

```
tfg-how-common-are-ia-images-on-web/
├── README.md
├── requirements.txt
├── configs/
│   └── spai.yaml                          # SPAI model configuration (ViT backbone, freq masking)
├── data/
│   ├── websites-list.csv                  # 25 websites across 5 sectors
│   └── websites-news.csv                  # 5 news-only websites (for Wayback Machine)
├── spai/                                  # SPAI model package
│   ├── models/                            # SID, MFM, Swin/ViT backbones, spectral filters
│   ├── data/                              # Dataset loaders, augmentations
│   ├── weights/spai.pth                   # Pre-trained checkpoint
│   └── ...
├── scraping/                              # Image collection tools
│   ├── advanced_image_scraper.py          # Main web crawler (BS4 + Playwright)
│   ├── wayback_machine.py                 # Wayback Machine CDX API scraper
│   ├── batch_scraper.py                   # Batch orchestrator for CSV lists
│   ├── run_wayback_full_scrape.py         # Batch orchestrator for Wayback Machine
│   ├── run_scraper_websites_list.sh       # SLURM script: batch scraping
│   └── run_wayback_scraper.sh             # SLURM script: wayback scraping
├── classification/                        # AI image classification
│   ├── inference.py                       # SPAI EndpointHandler (single image)
│   ├── infer_web_classifier.py            # Batch web image classifier
│   ├── infer_hf_dataset.py                # Evaluation on HF datasets
│   ├── eval_balanced_thresholds.py        # Threshold selection
│   ├── run_classify_wayback_years.sh      # SLURM: classify wayback images
│   ├── run_classify_websites_execution.sh # SLURM: classify per website
│   └── run_full_classification_pipeline.sh
├── analysis/                              # Statistical analysis
│   ├── analyze_zone_vs_ai.py              # Zone vs AI rate (chi-squared test)
│   ├── analyze_format_vs_ai.py            # Format vs AI rate
│   ├── combine_wayback_results.py         # Temporal AI rate aggregation
│   └── categorize_article_content_images.py
├── tools/                                 # Classification and plotting utilities
│   ├── classify_crawl_complete.py         # Classify all images from a crawl
│   ├── classify_wayback_years.py          # Classify wayback images by year
│   ├── classify_websites_execution.py     # Per-website classification
│   ├── plot_results_metrics.py            # Generate result figures
│   └── ...
└── tests/                                 # Unit tests for SPAI
```

---

## Installation

### 1. Create conda environment

```bash
conda create -n spai-hf-2 python=3.10
conda activate spai-hf-2
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -r scraping/requirements_scraper.txt
```

### 3. Install Playwright browsers (needed for JS-heavy websites)

```bash
playwright install chromium
```

---

## Input Data

### `data/websites-list.csv`

Contains 25 websites across 5 sectors. Used by the web scraper.

| Sector | Websites |
|--------|----------|
| news | La Vanguardia, RTVE, BBC News, Euronews, El Mundo |
| arts_illustration | Public Domain Review, DeviantArt, Rawpixel, ArtStation, Old Book Illustrations |
| education | UAB, MIT, Loyola, NDSU, UPC |
| e_commerce | IKEA, SHEIN, Sotheby's, MediaMarkt, Amazon |
| tourism_travel | France.fr, Spain.info, VisitBritain, Italia.it, Lonely Planet |

CSV columns: `url`, `sector`, `subsector`, `organization_name`

### `data/websites-news.csv`

Contains only the 5 news websites. Used by the Wayback Machine scraper.

---

## 1. Web Scraping

The scraper collects images from live websites using BeautifulSoup for static HTML and Playwright as fallback for JavaScript-rendered pages.

### What it does

- Crawls pages within the same domain (BFS)
- Extracts images from `<img>`, `<picture>`, `<source>`, `<amp-img>`, `<video poster>`, `<noscript>`, and CSS `background-image`
- Supports 15 lazy-loading attributes (`data-src`, `data-original`, `data-bg`, etc.)
- Picks the highest resolution from `srcset`
- Scrolls pages automatically in Playwright to trigger lazy/infinite loading
- Deduplicates via content hash (MD5) + perceptual hash
- Filters: minimum 80x80px, 3KB, formats JPEG/PNG/WebP/GIF
- Records metadata per image: URL, HTML tag, parent tag, CSS classes, zone classification
- Dismisses cookie banners automatically (21 selectors, multiple languages)

### Execution (SLURM cluster)

From the `scraping/` directory:

```bash
# All 25 websites
CSV_PATH=data/websites-list.csv \
BASE_OUTPUT=scrape_results \
MAX_PAGES=300 \
MIN_IMAGES_PER_PAGE=150 \
MAX_IMAGES_PER_SITE=200 \
DELAY=0.8 \
sbatch run_scraper_websites_list.sh --use-playwright-fallback --accept-cookies
```

```bash
# Single website (e.g. site 3 = BBC News)
CSV_PATH=data/websites-list.csv \
BASE_OUTPUT=scrape_results \
MAX_PAGES=300 \
MIN_IMAGES_PER_PAGE=150 \
MAX_IMAGES_PER_SITE=200 \
DELAY=0.8 \
CRAWL_SITE=3 \
sbatch run_scraper_websites_list.sh --use-playwright-fallback --accept-cookies
```

### Execution (local / without SLURM)

```bash
python scraping/batch_scraper.py \
    --csv data/websites-list.csv \
    --output-dir scrape_results \
    --max-images 200 \
    --max-pages 300 \
    --min-images-per-page 150 \
    --delay 0.8 \
    --use-playwright-fallback \
    --accept-cookies
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CSV_PATH` | `data/websites-list.csv` | CSV with `url` and `organization_name` columns |
| `BASE_OUTPUT` | `batch_scrape_results` | Output directory |
| `MAX_PAGES` | 5 | Max pages to crawl per website |
| `MIN_IMAGES_PER_PAGE` | 0 | Min images before triggering Playwright fallback |
| `MAX_IMAGES_PER_SITE` | 200 | Max images to download per website |
| `DELAY` | 1.0 | Delay between requests (seconds) |
| `CRAWL_SITE` | _(all)_ | 1-based index to scrape only one site |
| `--use-playwright-fallback` | off | Enable Playwright for JS-heavy sites |
| `--accept-cookies` | off | Auto-dismiss cookie banners |

### Output structure

```
scrape_results/run_YYYYMMDD_HHMMSS/
├── batch_summary.json
├── sites/
│   ├── 001_lavanguardia-com__la-vanguardia/
│   │   ├── images/                    # Downloaded image files
│   │   └── metadata/
│   │       └── images_metadata.csv    # Per-image metadata
│   ├── 002_rtve-es__rtve-noticias/
│   └── ...
```

### Metadata columns (`images_metadata.csv`)

| Column | Description |
|--------|-------------|
| `filename` | Image filename (MD5 hash + extension) |
| `image_url` | Source URL of the image |
| `page_url` | Page where the image was found |
| `html_tag` | HTML tag (`img`, `source`, `video[poster]`, etc.) |
| `parent_tag` | Parent HTML element |
| `classes` | CSS classes of the element and its parent |
| `element_id` | HTML id attribute |
| `width`, `height` | Image dimensions in pixels |
| `file_size` | File size in bytes |
| `format` | Image format (jpeg, png, webp, gif) |
| `content_hash` | MD5 hash of the file content |
| `perceptual_hash` | Perceptual hash for near-duplicate detection |
| `zone` | Page zone classification (header_nav, hero_banner, article_content, ad_sponsored, etc.) |

---

## 2. Wayback Machine Scraping

Collects historical snapshots of websites from the Internet Archive (2020-2025) using the CDX API.

### Execution (SLURM cluster)

From the `scraping/` directory:

```bash
# News websites only (2020-2025)
CSV_PATH=data/websites-news.csv \
MAX_IMAGES_PER_YEAR=200 \
MAX_SNAPSHOTS=100 \
sbatch run_wayback_scraper.sh
```

### Execution (local)

```bash
python scraping/run_wayback_full_scrape.py \
    --csv data/websites-news.csv \
    --output wayback_output \
    --start-year 2020 \
    --end-year 2025 \
    --max-snapshots-per-year 100 \
    --max-images-per-year 200
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CSV_PATH` | `data/websites-news-arts.csv` | CSV with `url` column |
| `OUTPUT_DIR` | `wayback_images_by_year/output` | Output directory |
| `START_YEAR` | 2020 | First year to query |
| `END_YEAR` | 2025 | Last year to query |
| `MAX_SNAPSHOTS` | 5 | Max CDX snapshots to fetch per year |
| `MAX_IMAGES_PER_YEAR` | 50 | Max images to download per year per site |
| `DELAY` | 1.5 | Delay between Wayback Machine requests (seconds) |
| `CRAWL_SITE` | _(all)_ | 1-based index for a single site |

### Output structure

```
wayback_output/
├── wayback_summary.json
└── sites/
    ├── bbc-co-uk/
    │   ├── 2020/    # Images from 2020 snapshots
    │   ├── 2021/
    │   ├── ...
    │   ├── 2025/
    │   └── manifest.csv
    ├── lavanguardia-com/
    └── ...
```

---

## 3. SPAI Classification

The SPAI classifier scores images from 0 (real) to 1 (AI-generated). Threshold: **0.35**.

### Single image

```bash
python classification/inference.py \
    --image "/path/to/image.jpg" \
    --model-dir /home/aaasifar/spai-hf
```

Output: `{score: 0.82, predicted_label: 1, predicted_label_name: "ai", threshold: 0.5}`

### Classify all scraped images (by sector)

```bash
python tools/classify_crawl_complete.py \
    --crawl-dir scrape_results/run_YYYYMMDD_HHMMSS/ \
    --output-dir classification_results/ \
    --model-dir /home/aaasifar/spai-hf \
    --threshold 0.35
```

### Classify per website

```bash
python tools/classify_websites_execution.py \
    --images-dir scrape_results/run_YYYYMMDD_HHMMSS/sites/001_lavanguardia-com__la-vanguardia/images/ \
    --output-dir classification_results/lavanguardia/ \
    --model-dir /home/aaasifar/spai-hf \
    --threshold 0.35
```

### Classify Wayback Machine images by year

```bash
python tools/classify_wayback_years.py \
    --root-dir wayback_output/sites/bbc-co-uk/ \
    --output-dir wayback_classification/bbc/ \
    --model-dir /home/aaasifar/spai-hf \
    --threshold 0.35
```

### SLURM scripts

```bash
# Classify all wayback images
sbatch classification/run_classify_wayback_years.sh

# Classify all scraped website images
sbatch classification/run_classify_websites_execution.sh

# Full pipeline (scrape + classify + analyze)
sbatch classification/run_full_classification_pipeline.sh
```

---

## 4. Analysis

### Zone vs AI rate (chi-squared test)

```bash
python analysis/analyze_zone_vs_ai.py --results-dir classification_results/
```

### Format vs AI rate

```bash
python analysis/analyze_format_vs_ai.py --results-dir classification_results/
```

### Temporal analysis (Wayback Machine)

```bash
python analysis/combine_wayback_results.py --results-dir wayback_classification/
```

### Generate result plots

```bash
python tools/plot_results_metrics.py --results-dir classification_results/
```

---

## SLURM Configuration

All SLURM scripts use:

| Setting | Value |
|---------|-------|
| Partition | `pg2tfg12` |
| QoS | `q_pg2tfg12` |
| CPUs | 2 |
| Memory | 8G (scraper), 4G (others) |
| Time limit | 6h (scraper), 4h (others) |
| Conda env | `spai-hf-2` |
| Logs | `/home/aaasifar/spai-hf/scraping_logs/` |

---

## File Descriptions

### `scraping/`

| File | Description |
|------|-------------|
| `advanced_image_scraper.py` | Main crawler: BS4 + Playwright fallback, 15 lazy-load attributes, auto-scroll, cookie dismissal, zone classification |
| `wayback_machine.py` | Wayback Machine scraper via CDX API, year-based image collection |
| `batch_scraper.py` | Batch orchestrator: reads CSV, creates per-site output folders, runs scraper for each site |
| `run_wayback_full_scrape.py` | Batch orchestrator for Wayback Machine scraping |
| `run_scraper_websites_list.sh` | SLURM script for batch website scraping |
| `run_wayback_scraper.sh` | SLURM script for Wayback Machine scraping |
| `extract_article_content_images.py` | Extracts images from article body content only |
| `create_ai_images_folder.py` | Collects all AI-classified images into a single folder |

### `classification/`

| File | Description |
|------|-------------|
| `inference.py` | SPAI EndpointHandler: scores a single image (URL, path, base64, or PIL) |
| `infer_web_classifier.py` | Batch classifier for scraped web images with zone/site metadata |
| `infer_hf_dataset.py` | Evaluates SPAI on Hugging Face datasets (GenAI-Bench) |
| `eval_balanced_thresholds.py` | Threshold evaluation on balanced datasets, accuracy curves |

### `analysis/`

| File | Description |
|------|-------------|
| `analyze_zone_vs_ai.py` | Chi-squared test of zone vs AI classification |
| `analyze_format_vs_ai.py` | AI rate by image format (JPEG, PNG, WebP) |
| `combine_wayback_results.py` | Aggregates temporal classification into annual AI rates |
| `categorize_article_content_images.py` | Categorizes article images by content type |

### `tools/`

| File | Description |
|------|-------------|
| `classify_crawl_complete.py` | Runs SPAI on all images from a crawl directory |
| `classify_wayback_years.py` | Runs SPAI on Wayback Machine images by year |
| `classify_websites_execution.py` | Per-website classification orchestrator |
| `plot_results_metrics.py` | Generates result figures (histograms, bars, confusion matrices) |
| `plot_style.py` | Shared matplotlib styling |

### `spai/`

| Component | Description |
|-----------|-------------|
| `models/` | SID (Spectral Image Detector), MFM, Swin/ViT backbones, spectral filters, frequency losses |
| `data/` | Dataset loaders, augmentation pipelines, LMDB storage |
| `weights/spai.pth` | Pre-trained checkpoint (CVPR 2025) |
| `configs/spai.yaml` | Model configuration: ViT-Base, 12 layers, patch size 224, frequency masking radius 16 |

---

## Citation

```
@inproceedings{karageorgiou2025any,
  title={Any-resolution ai-generated image detection by spectral learning},
  author={Karageorgiou, Dimitrios and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18706--18717},
  year={2025}
}
```

## License

Source code is licensed under Apache 2.0.
Third-party datasets and dependencies keep their own licenses.
