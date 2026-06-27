# How Common Are AI-Generated Images on the Web?

Automated pipeline to measure the prevalence of AI-generated images across
publicly accessible websites. It combines a web image scraper, a Wayback Machine
historical scraper, and the SPAI spectral classifier to detect AI-generated
content at scale.

The pipeline was applied to 25 websites across five sectors (news and media, art
and illustration, education, e-commerce, and tourism and travel), producing a
dataset of ~4,500 classified images.

## Requirements

- Python 3.11+
- CUDA-capable GPU (compute capability >= 7.5)
- The [SPAI model](https://github.com/mever-team/spai) weights at `spai/weights/spai.pth`

### Installation

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt

# Scraping dependencies (Playwright, BeautifulSoup, etc.)
pip install -r scraping/requirements_scraper.txt
playwright install chromium
```

## Execution Steps

The pipeline has three main stages: **scraping**, **classification**, and
**analysis**. Each stage can be run independently.

### 1. Scraping: Collect Images from Websites

The scraper uses BeautifulSoup for static content and Playwright for
JS-rendered pages. It extracts images from HTML tags, CSS background
properties, and semantic meta tags. Filtering removes duplicates (content +
perceptual hashing), images below 100x100 px or 5 KB, and non-JPEG/PNG/WebP
formats.

```bash
# Scrape all 25 websites from the CSV list
python scraping/run_csv_full_scrape.py \
    --csv data/websites-list.csv \
    --output results/batch_scrape_results/

# Scrape a single website
python scraping/advanced_image_scraper.py \
    --url https://www.example.com \
    --output results/batch_scrape_results/example/
```

On a SLURM cluster:
```bash
sbatch scraping/run_scraper_websites_list.sh
```

### 2. Scraping: Wayback Machine (Historical Snapshots)

Collects archived snapshots from the Wayback Machine (2020-2025) using the
CDX API, up to 50 images per website per year. Used for temporal trend
analysis.

```bash
# Scrape historical snapshots for news and arts websites
python scraping/run_wayback_full_scrape.py \
    --csv data/websites-news-arts.csv \
    --output results/wayback_images_by_year/
```

On a SLURM cluster:
```bash
sbatch scraping/run_wayback_scraper.sh
```

### 3. Classification: Run the SPAI Classifier

SPAI is a spectral learning classifier that assigns each image a score
between 0 (real) and 1 (AI-generated). The decision threshold is **0.35**.

#### Classify a Single Image

```bash
python classification/inference.py \
    --image /path/to/image.jpg \
    --model-dir .
```

#### Classify All Scraped Images (by Category)

Runs classification on all images organized by category folders, generates
per-category statistics and plots:

```bash
python tools/classify_crawl_complete.py \
    --crawl-dir results/batch_scrape_results/ \
    --output-dir results/classification_results/ \
    --model-dir . \
    --threshold 0.35
```

#### Classify Images from a Metadata CSV

Takes a CSV with image paths/URLs and metadata (sector, organization, etc.)
and produces detailed breakdowns by sector, website, and image type:

```bash
python tools/classify_websites_execution.py \
    --metadata-csv metadata.csv \
    --output-dir results/classification_results/ \
    --model-dir . \
    --threshold 0.35
```

Or classify directly from a directory of images:

```bash
python tools/classify_websites_execution.py \
    --images-dir results/batch_scrape_results/ \
    --output-dir results/classification_results/ \
    --model-dir . \
    --threshold 0.35
```

#### Classify Wayback Machine Images by Year

Expects a root directory with year subfolders (e.g. `2020/`, `2021/`, ...),
classifies each year separately and generates temporal comparison plots:

```bash
python tools/classify_wayback_years.py \
    --root-dir results/wayback_images_by_year/site_name/ \
    --output-dir results/wayback_classification_results/site_name/ \
    --model-dir . \
    --threshold 0.35
```

On a SLURM cluster (classifies all sites at once):
```bash
sbatch classification/run_classify_wayback_years.sh
sbatch classification/run_classify_websites_execution.sh
```

#### Evaluate on Hugging Face Datasets

Run SPAI against benchmark datasets (e.g. GenAI-Bench, AI-vs-Real):

```bash
python classification/infer_hf_dataset.py \
    --dataset Parveshiiii/AI-vs-Real \
    --split train \
    --image-column image \
    --max-images 200 \
    --threshold 0.35 \
    --output-csv results/balanced_eval/results.csv
```

#### Evaluate Threshold Selection

Test multiple thresholds on balanced datasets to find the optimal operating
point:

```bash
python classification/eval_balanced_thresholds.py \
    --dataset results/balanced_eval/ \
    --output-dir results/balanced_eval/
```

### 4. Analysis

#### Zone Analysis (Page Zones vs AI Rate)

Analyzes whether AI-generated images concentrate in specific page zones
(article content, navigation, sidebar, ads, etc.) using chi-squared tests:

```bash
python analysis/analyze_zone_vs_ai.py \
    --results-dir results/classification_results/
```

#### Temporal Analysis (AI Rate Over Time)

Combines per-site Wayback Machine classification results into global
year-by-year trends:

```bash
python analysis/combine_wayback_results.py \
    --root results/wayback_classification_results/
```

#### Format Analysis

Analyzes AI rate by image format (JPEG, PNG, WebP):

```bash
python analysis/analyze_format_vs_ai.py
```

#### Article Content Type Analysis

Categorizes article images by content type (news, feature, blog, commercial):

```bash
python analysis/categorize_article_content_images.py
```

### 5. Generate Plots

Regenerate all figures from existing classification results:

```bash
python tools/generate_existing_results_figures.py
python tools/plot_results_metrics.py
python tools/model_analysis_graphs.py
```

## Full Pipeline (End to End)

To reproduce the complete study:

```bash
# 1. Scrape images from the 25 websites
python scraping/run_csv_full_scrape.py \
    --csv data/websites-list.csv \
    --output results/batch_scrape_results/

# 2. Scrape Wayback Machine snapshots (2020-2025)
python scraping/run_wayback_full_scrape.py \
    --csv data/websites-news-arts.csv \
    --output results/wayback_images_by_year/

# 3. Classify all scraped images
python tools/classify_crawl_complete.py \
    --crawl-dir results/batch_scrape_results/ \
    --output-dir results/classification_results/ \
    --model-dir . --threshold 0.35

# 4. Classify Wayback Machine images by year
python tools/classify_wayback_years.py \
    --root-dir results/wayback_images_by_year/ \
    --output-dir results/wayback_classification_results/ \
    --model-dir . --threshold 0.35

# 5. Run zone analysis
python analysis/analyze_zone_vs_ai.py \
    --results-dir results/classification_results/

# 6. Run temporal analysis
python analysis/combine_wayback_results.py \
    --root results/wayback_classification_results/
```

On the UAB SLURM cluster, use the `.sh` scripts in `scraping/` and
`classification/` instead.

## Project Structure

```text
├── configs/
│   └── spai.yaml                       # SPAI model configuration
│
├── classification/                     # AI image classification
│   ├── inference.py                    # SPAI EndpointHandler (single image)
│   ├── infer_web_classifier.py         # Crawl + classify from URL
│   ├── infer_hf_dataset.py             # Evaluate on HF datasets
│   ├── eval_balanced_thresholds.py     # Threshold optimization
│   ├── run_classify_wayback_years.sh   # SLURM: wayback classification
│   ├── run_classify_websites_execution.sh  # SLURM: website classification
│   ├── run_full_classification_pipeline.sh # SLURM: end-to-end pipeline
│   └── run_hf_ai_vs_real_200_live.sh   # SLURM: mixed-data validation
│
├── analysis/                           # Statistical analysis
│   ├── analyze_zone_vs_ai.py           # Page zone vs AI rate (chi-squared)
│   ├── analyze_format_vs_ai.py         # Image format vs AI rate
│   ├── combine_wayback_results.py      # Temporal trend aggregation
│   ├── combine_wayback_manifests.py    # Wayback manifest combination
│   ├── categorize_article_content_images.py  # Content type breakdown
│   └── cross_reference_article_content_predictions.py
│
├── tools/                              # Execution and visualization
│   ├── classify_crawl_complete.py      # Batch classification by category
│   ├── classify_wayback_years.py       # Wayback classification by year
│   ├── classify_websites_execution.py  # Per-website classification from CSV
│   ├── classify_news_from_crawl.py     # News image classification
│   ├── analyze_crawl_results.py        # Summary statistics generation
│   ├── plot_results_metrics.py         # Result figure generation
│   ├── plot_style.py                   # Shared matplotlib styling
│   ├── plot_crawl_category_progress.py # Category progress plots
│   ├── merge_genaibench_and_local.py   # Merge benchmark datasets
│   ├── model_analysis_graphs.py        # Model performance graphs
│   └── generate_existing_results_figures.py  # Regenerate all figures
│
├── data/                               # Input data
│   ├── websites-list.csv               # 25 websites across 5 sectors
│   └── websites-news-arts.csv          # News + art subset for Wayback
│
├── scraping/                           # Image collection
│   ├── advanced_image_scraper.py       # Main crawler (BS4 + Playwright)
│   ├── wayback_machine.py              # Wayback Machine CDX API scraper
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
├── spai/                               # SPAI model (Karageorgiou et al.)
│   ├── models/                         # SID, MFM, Swin/ViT backbones
│   ├── data/                           # Dataset loaders, augmentations
│   └── weights/spai.pth                # Pre-trained checkpoint
│
├── results/                            # Output directory
│   ├── batch_scrape_results/           # Scraped images by website
│   ├── classification_results/         # Per-image SPAI predictions
│   ├── wayback_images_by_year/         # Wayback Machine images (2020-2025)
│   ├── article_content_images/         # Article body images
│   ├── balanced_eval/                  # Threshold evaluation data
│   └── results_plots/                  # Generated figures
│
├── docs/
│   └── PIPELINE_STEPS.md
│
└── requirements.txt
```

## Citation

If you use SPAI, please cite the original work:

```bibtex
@inproceedings{karageorgiou2025any,
  title     = {Any-resolution AI-generated image detection by spectral learning},
  author    = {Karageorgiou, Dimitrios and Papadopoulos, Symeon and
               Kompatsiaris, Ioannis and Gavves, Efstratios},
  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages     = {18706--18717},
  year      = {2025}
}
```

## License

Source code is licensed under Apache 2.0.
Third-party datasets and dependencies keep their own licenses.
