# How Common Are AI-Generated Images on the Web?

**TFG en Enginyeria de Dades -- Universitat Autonoma de Barcelona (UAB), 2025/26**

Amina Aasifar El Ouahabi  
Supervised by Alexandra Gomez Villa (Computer Vision Center)

This repository contains all the code and data for the study of AI-generated
image prevalence across publicly accessible websites. It includes an automated
web image scraper, a Wayback Machine historical scraper, the SPAI spectral
classifier, and the resulting dataset of 4,520 classified images from 25
websites across five sectors: news and media, art and illustration, education,
e-commerce, and tourism and travel.

Key findings: a global AI rate of 19.2%, led by the news and media sector at
24.7%, with a pronounced concentration in article content zones (28.6%) and a
temporal adoption peak around 2022.

## Project Structure

```text
spai-hf/
├── README.md
├── requirements.txt
│
├── configs/
│   └── spai.yaml                       # SPAI model configuration
│
├── spai/                               # SPAI model package (CVPR 2025)
│   ├── data/                           # Dataset loaders, augmentations, LMDB
│   ├── models/                         # SID, MFM, Swin/ViT backbones, losses
│   ├── tools/                          # Dataset CSV generators
│   ├── weights/spai.pth                # Trained checkpoint
│   └── ...
│
├── scraping/                           # Image collection (Sections 4.2, 4.4)
│   ├── advanced_image_scraper.py       # Main web crawler
│   ├── wayback_machine.py              # Wayback Machine scraper
│   ├── batch_scraper.py                # Batch orchestrator
│   ├── run_csv_full_scrape.py          # Batch scraping launcher
│   ├── run_wayback_full_scrape.py      # Wayback scraping launcher
│   ├── extract_article_content_images.py
│   ├── create_ai_images_folder.py
│   ├── run_scraper.sh                  # SLURM job scripts
│   ├── run_scraper_websites_list.sh
│   ├── run_wayback_scraper.sh
│   └── requirements_scraper.txt
│
├── classification/                     # AI image classification (Section 4.3)
│   ├── inference.py                    # SPAI EndpointHandler
│   ├── infer_web_classifier.py         # Batch web image classifier
│   ├── infer_hf_dataset.py             # GenAI-Bench evaluation
│   ├── eval_balanced_thresholds.py     # Threshold selection (Appendix A)
│   ├── run_classify_wayback_years.sh   # SLURM: wayback classification
│   ├── run_classify_websites_execution.sh
│   ├── run_full_classification_pipeline.sh
│   └── run_hf_ai_vs_real_200_live.sh   # Mixed-data validation (200 imgs)
│
├── analysis/                           # Statistical analysis (Sections 5, 6)
│   ├── analyze_zone_vs_ai.py           # Hypothesis H1: page zones
│   ├── analyze_format_vs_ai.py         # Format vs AI rate
│   ├── combine_wayback_results.py      # Temporal analysis (Section 5.5)
│   ├── combine_wayback_manifests.py    # Temporal data combination
│   ├── categorize_article_content_images.py  # Content type analysis (Fig 13)
│   └── cross_reference_article_content_predictions.py
│
├── tools/                              # Execution and plotting utilities
│   ├── classify_crawl_complete.py      # Main classification execution
│   ├── classify_wayback_years.py       # Wayback classification by year
│   ├── classify_websites_execution.py  # Per-website classification
│   ├── classify_news_from_crawl.py     # News image classification
│   ├── analyze_crawl_results.py        # Results summary statistics
│   ├── plot_results_metrics.py         # Generates paper figures
│   ├── plot_style.py                   # Shared matplotlib style
│   ├── merge_genaibench_and_local.py   # Merges benchmark datasets
│   ├── model_analysis_graphs.py        # Model performance graphs
│   └── generate_existing_results_figures.py
│
├── tests/                              # Unit tests for SPAI
│   ├── data/
│   └── models/
│
├── data/                               # Input: website lists
│   ├── websites-list.csv               # 25 websites, 5 sectors
│   └── websites-news-arts.csv          # Subset for Wayback Machine
│
├── results/                            # Output: dataset and results
│   ├── batch_scrape_results/           # Scraped images (~4,500 images)
│   ├── classification_results/         # Per-image SPAI predictions (CSVs)
│   ├── wayback_images_by_year/         # Wayback Machine images (2020-2025)
│   ├── article_content_images/         # Article body images
│   ├── balanced_eval/                  # Threshold evaluation data
│   ├── results_plots/                  # Generated figures
│   ├── results.csv                     # All classification results
│   └── results.jsonl                   # Same in JSONL format
│
└── docs/
    └── PIPELINE_STEPS.md               # Pipeline documentation
```

## Methodology Overview

The project follows a four-stage pipeline, each corresponding to a section of
the paper:

### 1. Website Selection (Section 4.1)

25 websites were selected across five sectors. The full list is in
`data/websites-list.csv` with columns: `url`, `sector`, `subsector`,
`organization_name`.

### 2. Image Collection (Sections 4.2, 4.4)

Images were collected using `scraping/advanced_image_scraper.py`, which
combines BeautifulSoup for static content and Playwright for JS-rendered pages.
The crawler extracts images from HTML tags, CSS background properties, and
semantic meta tags.

Filtering: duplicates removed via content + perceptual hashing, images below
100x100px or 5KB excluded, only JPEG/PNG/WebP retained.

For the temporal analysis, `scraping/wayback_machine.py` collects archived
snapshots from the Wayback Machine (2020-2025) using the CDX API.

**Run scraping:**
```bash
# Current websites
python scraping/run_csv_full_scrape.py --csv data/websites-list.csv --output results/batch_scrape_results/

# Wayback Machine (historical)
python scraping/run_wayback_full_scrape.py --csv data/websites-news-arts.csv --output results/wayback_images_by_year/
```

### 3. Classification (Section 4.3)

Images are classified using SPAI, a spectral learning classifier that assigns
each image a score between 0 (real) and 1 (AI-generated). The operational
threshold is **0.35**, selected through three evaluation stages (see Appendix A
of the paper).

**Run classification:**
```bash
# Single image
python classification/inference.py --image "/path/to/image.jpg" --model-dir .

# All scraped images
python tools/classify_crawl_complete.py \
    --crawl-dir results/batch_scrape_results/ \
    --output-dir results/classification_results/ \
    --model-dir . --threshold 0.35
```

### 4. Analysis (Sections 5, 6)

- **Sector-level (Hc):** AI rate compared across five sectors using weighted rates
- **Zone-level (H1):** `analysis/analyze_zone_vs_ai.py` -- chi-squared test on zone x class contingency table
- **Temporal:** `analysis/combine_wayback_results.py` -- AI rate per year for news websites (2020-2025)
- **Content type:** `analysis/categorize_article_content_images.py` -- breakdown by article content type (Fig 13)

## File Descriptions

### `scraping/` -- Image Collection

| File | Paper ref | Description |
|------|-----------|-------------|
| `advanced_image_scraper.py` | Sec 4.2 | Main crawler: BeautifulSoup + Playwright, filters by size/format/zone, extracts metadata (HTML tag, parent, CSS class). |
| `wayback_machine.py` | Sec 4.4 | Wayback Machine scraper: CDX API, collects archived snapshots by year (2020-2025), up to 50 images/website/year. |
| `batch_scraper.py` | Sec 4.2 | Batch orchestrator: runs the crawler across all websites in a CSV list. |
| `run_csv_full_scrape.py` | | Python launcher for a full batch scrape from CSV. |
| `run_wayback_full_scrape.py` | | Python launcher for Wayback Machine scrape. |
| `extract_article_content_images.py` | Sec 6.2 | Extracts images from article body content, filtering out navigation/ads/sidebars. |
| `create_ai_images_folder.py` | | Collects all AI-classified images into a single folder, organized by website. |
| `run_scraper.sh` | | SLURM job script for single-site scraping. |
| `run_scraper_websites_list.sh` | | SLURM job script for batch scraping. |
| `run_wayback_scraper.sh` | | SLURM job script for Wayback Machine scraping. |
| `requirements_scraper.txt` | | Python dependencies for scraping (Playwright, BeautifulSoup, etc.). |

### `classification/` -- SPAI Classification

| File | Paper ref | Description |
|------|-----------|-------------|
| `inference.py` | Sec 4.3 | SPAI EndpointHandler: scores images 0-1, supports URL/path/base64/PIL input. |
| `infer_web_classifier.py` | Sec 4.3 | Batch classifier for scraped web images, outputs per-image predictions with zone/site metadata. |
| `infer_hf_dataset.py` | Sec 4.3 | Evaluates SPAI on Hugging Face datasets (GenAI-Bench, AI-vs-Real). |
| `eval_balanced_thresholds.py` | App A | Evaluates thresholds on balanced datasets, produces accuracy curves and confusion matrices. |
| `run_classify_wayback_years.sh` | Sec 4.4 | SLURM: classifies Wayback Machine images by year. |
| `run_classify_websites_execution.sh` | | SLURM: classification pipeline per website. |
| `run_full_classification_pipeline.sh` | | SLURM: end-to-end pipeline (crawl + classify + analyze). |
| `run_hf_ai_vs_real_200_live.sh` | Sec 4.3 | SLURM: mixed-data validation (120 AI + 80 real, threshold = 0.35, accuracy = 82.5%). |

### `analysis/` -- Statistical Analysis

| File | Paper ref | Description |
|------|-----------|-------------|
| `analyze_zone_vs_ai.py` | Sec 5.4 | Tests H1: chi-squared test of independence on zone x class table, two-proportion test for Article content vs rest. |
| `analyze_format_vs_ai.py` | | Analyzes AI rate by image format (JPEG, PNG, WebP). |
| `combine_wayback_results.py` | Sec 5.5 | Aggregates temporal classification results into annual AI rates (Table 1, Fig 11). |
| `combine_wayback_manifests.py` | Sec 5.5 | Combines Wayback Machine download manifests across sites and years. |
| `categorize_article_content_images.py` | Sec 6.2 | Categorizes article images by content type: news, feature, blog, commercial (Fig 13). |
| `cross_reference_article_content_predictions.py` | Sec 6.2 | Cross-references AI predictions with source article context. |

### `tools/` -- Execution Utilities

| File | Description |
|------|-------------|
| `classify_crawl_complete.py` | Runs SPAI on all images from a crawl directory, saves per-site CSVs. |
| `classify_wayback_years.py` | Runs SPAI on Wayback Machine images organized by year. |
| `classify_websites_execution.py` | Orchestrates classification across multiple websites. |
| `classify_news_from_crawl.py` | Classifies only news-section images from a crawl. |
| `analyze_crawl_results.py` | Generates summary statistics from classification CSVs. |
| `plot_results_metrics.py` | Generates paper figures: score histograms, AI rate bars, confusion matrices. |
| `plot_style.py` | Shared matplotlib styling for consistent figure appearance. |
| `merge_genaibench_and_local.py` | Merges GenAI-Bench with locally scraped data for threshold evaluation. |
| `model_analysis_graphs.py` | Model performance analysis graphs (accuracy by model, FN rates). |
| `generate_existing_results_figures.py` | Regenerates all figures from existing result CSVs. |

### `data/` -- Input Data

| File | Paper ref | Description |
|------|-----------|-------------|
| `websites-list.csv` | Sec 4.1 | 25 websites across 5 sectors. Columns: `url`, `sector`, `subsector`, `organization_name`. |
| `websites-news-arts.csv` | Sec 4.4 | News + art subset used for Wayback Machine temporal analysis. |

### `results/` -- Dataset and Classification Results

| Directory / File | Paper ref | Description |
|------------------|-----------|-------------|
| `batch_scrape_results/` | Sec 4.2 | Raw scraped images organized by run date and website (~4,500 images). |
| `classification_results/` | Sec 5 | Per-image SPAI predictions organized by website and page type. Each CSV contains: `score`, `predicted_label`, `is_ai`, `organization_name`, `sector`, `page_url`, `image_url`, `image_type`. |
| `wayback_images_by_year/` | Sec 4.4 | Historical images from Wayback Machine snapshots (2020-2025). |
| `article_content_images/` | Sec 6.2 | Images extracted from article body content zones. Subfolder `noticias_ia/` contains AI-related news article images. |
| `balanced_eval/` | App A | Threshold evaluation: `genai_b120_scores.csv` (per-image scores from 6 generators), `genai_b120_thresholds_global.csv` (accuracy per threshold). |
| `results_plots/` | | Generated figures: `score_histogram.png`, `confusion_matrix.png`, `real_vs_ai_bars.png`, `results_overview.png`, accuracy by model plots. |
| `results.csv` | Sec 5 | Main classification results aggregated across all websites. |
| `results.jsonl` | | Same data in JSONL format with full metadata per image. |

### `spai/` -- SPAI Model (Karageorgiou et al., CVPR 2025)

| Component | Description |
|-----------|-------------|
| `models/` | Neural architectures: SID (Spectral Image Detector), MFM (Masked Frequency Modeling), Swin Transformer, ViT backbones, spectral filters, frequency losses. |
| `data/` | Dataset loaders, readers, data augmentation pipelines, LMDB file storage. |
| `weights/spai.pth` | Pre-trained checkpoint used for all classifications in this study. |
| `config.py` | YACS configuration system. |
| `main_mfm.py` | MFM pre-training entry point. |

## Installation

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r scraping/requirements_scraper.txt
```

## Reproducing the Study

```bash
# 1. Scrape images from the 25 websites
python scraping/run_csv_full_scrape.py --csv data/websites-list.csv --output results/batch_scrape_results/

# 2. Scrape Wayback Machine snapshots (2020-2025)
python scraping/run_wayback_full_scrape.py --csv data/websites-news-arts.csv --output results/wayback_images_by_year/

# 3. Classify all scraped images with SPAI (threshold = 0.35)
python tools/classify_crawl_complete.py --crawl-dir results/batch_scrape_results/ --output-dir results/classification_results/ --model-dir . --threshold 0.35

# 4. Classify Wayback Machine images by year
python tools/classify_wayback_years.py --input-dir results/wayback_images_by_year/ --output-dir results/wayback_classification_results/ --model-dir . --threshold 0.35

# 5. Run zone analysis (H1)
python analysis/analyze_zone_vs_ai.py --results-dir results/classification_results/

# 6. Run temporal analysis
python analysis/combine_wayback_results.py --results-dir results/wayback_classification_results/
```

On the UAB SLURM cluster, use the `.sh` scripts in `scraping/` and `classification/` instead.

## Citation

```text
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
