# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

An end-to-end, high-performance machine learning pipeline for multi-source business entity resolution, developed for the Amazon ML Challenge 2026. The system links noisy, fragmented business records across three independent data sources without shared identifiers, optimizing for precision-weighted macro $F_{0.5}$ score.

---

## Project Highlights & Impact

* **Large-Scale Data Processing:** Processed a massive **~1.1 GB compressed (~2.4 GB raw)** dataset comprising over **12.5 million business records** (2.2M reference entities, 5.0M Source 2 records, 5.2M Source 3 records) across multiple international geographic regions (**US, India, and France**).
* **High-Precision Multi-Strategy Blocking:** Reduced the intractable $\mathcal{O}(N \times M)$ pairwise comparison space to the **Top 60 highest-probability candidates** per reference entity using a 5-strategy ensemble, achieving high recall ceilings while maintaining a strictly bounded memory footprint.
* **Feature Engineering:** Extracted **25 distinct numeric similarity features** capturing fine-grained character-level, token-level, phonetic, and address structural overlaps.
* **Compliant Tree-Based Modeling:** Trained an optimized **LightGBM binary classifier** (500 estimators, 31 leaves) fully compliant with the competition's strict $\le 8\text{B}$ parameter constraint and MIT/Apache 2.0 open-source licensing.
* **Precision-Weighted Metric Optimization:** Programmatically tuned decision thresholds on hold-out validation splits to maximize the competition's **macro $F_{0.5}$ score** ($\sim 0.7741$ on full validation runs), weighting precision $1.25\times$ over recall to aggressively penalize false merges.
* **Strict Submission Compliance:** Output deliverables ([matching_results.tsv](output/matching_results.tsv) and [candidate_pairs.tsv](output/candidate_pairs.tsv)) passed the official submission validator with **100% compliance** across all **1,732,544 test entities** (`PASS - no blocking issues found`).

---

## Dataset & Challenge Metrics

### 1. Data Scale Breakdown

| Dataset Partition | File Name | Size (MB) | Record Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | `train_source1.tsv` | 200.3 MB | ~2,206,821 | Deduplicated reference business entity set |
| **Train Source 2** | `train_source2.tsv` | 466.6 MB | ~5,000,000 | Independent noisy source (zero, one, or many matches) |
| **Train Source 3** | `train_source3.tsv` | 480.4 MB | ~5,200,000 | Independent noisy source (zero, one, or many matches) |
| **Train Ground Truth**| `train_ground_truth.tsv` | 121.1 MB | 2,206,821 | True matching mappings across sources |
| **Test Source 1** | `test_source1.tsv` | 166.9 MB | 1,732,544 | Evaluation reference entities (required in submission) |
| **Test Source 2** | `test_source2.tsv` | 485.9 MB | ~5,100,000 | Evaluation test source 2 |
| **Test Source 3** | `test_source3.tsv` | 482.6 MB | ~5,200,000 | Evaluation test source 3 |
| **Total Pipeline** | **7 TSV Files** | **~2,403 MB** | **> 12,500,000** | **Complete challenge dataset** |

### 2. Ground Truth Distribution Insights
* **Total Reference Entities ($S_1$):** 2,206,821
* **Entities with $\ge 1$ Match:** 2,083,574 (94.42%)
* **Singletons (No Match):** 123,247 (5.58%) — Correctly predicting empty match lists scores 1.0; false merges on singletons score 0.0.
* **Maximum Matches for a Single Entity:** 11 matches across Source 2 and Source 3.

### 3. Evaluation Metric: Macro $F_{0.5}$
The evaluation uses the precision-heavy **$F_{\beta}$ score ($\beta = 0.5$)**:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

* **Macro-Averaging:** Computed independently per Source 1 entity, then averaged across **all** 1,732,544 Source 1 entities in the evaluation set.
* **Precision Weighting:** False merges (merging two distinct businesses) are penalized $4\times$ more severely than false negatives (missed matches).

---

## System Architecture

```
                      +---------------------------------------+
                      |   Source 1 (Reference Entity Set)     |
                      |   Source 2 & Source 3 (Noisy Sources) |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Multi-Strategy Normalization     |
                      |  - Unicode NFKC & Whitespace Collapse |
                      |  - 20+ Legal Suffixes & Abbreviations |
                      |  - Regex PIN/ZIP & Street Number Extr.|
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        Multi-Strategy Blocking        |
                      |  1. Token Inverted Index              |
                      |  2. Char TF-IDF KNN (Business Names)  |
                      |  3. Token TF-IDF KNN (Addresses)      |
                      |  4. Double Metaphone Phonetic Keying  |
                      |  5. Sorted Neighborhood (Name Prefix) |
                      +-------------------+-------------------+
                                          | (Top 60 Candidates / S1)
                                          v
                      +---------------------------------------+
                      |      25-Feature Engineering Engine    |
                      |  - String Distances (Lev, Jaro, Dice) |
                      |  - Token Overlap (Sort & Set Ratio)   |
                      |  - Address (PIN/ZIP, Street Numbers)  |
                      |  - Exact & Suffix-Stripped Matches    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       LightGBM Classifier Model       |
                      |  - Binary Objective (500 est, 31 leaf)|
                      |  - Strict <= 8B Parameter Constraint  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Precision-Weighted Decoding      |
                      |  - Threshold Optimization (0.05-0.95) |
                      |  - Direct Macro F_0.5 Tuning          |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         Final Deliverables (PASS)     |
                      |  - output/matching_results.tsv        |
                      |  - output/candidate_pairs.tsv         |
                      +-------------------+-------------------+
```

---

## Technical Pipeline Modules

### 1. Data Normalization & Preprocessing (`src/normalize.py`)
* **Text Standardization:** Applied Unicode NFKC normalization, lowercase canonicalization, whitespace collapsing, and punctuation stripping across all 12.5M records.
* **Regex Legal Suffix Cleaning:** Standardized 20+ global business entity legal types (e.g., `corp`, `corporation`, `ltd`, `limited`, `pvt`, `private`, `inc`, `incorporated`, `llc`, `gmbh`, `sa`, `co`).
* **Address Standardizer:** Canonicalized common address tokens (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `ste` $\rightarrow$ `suite`).
* **Entity Extraction:** Extracted leading numeric street numbers and 5-digit (US) / 6-digit (India) postal PIN/ZIP codes for targeted block filtering.

### 2. Candidate Generation / Blocking (`src/blocking.py`)
To make entity resolution tractable on millions of records, an ensemble of 5 blocking strategies is deployed:
1. **Token Inverted Index:** Inverted lookup indexing rare, informative unigrams and bigrams from business names.
2. **Character n-gram TF-IDF KNN:** Nearest-neighbor search over 3-4 character n-gram TF-IDF representations to catch typos, minor transpositions, and phonetic misspellings.
3. **Address TF-IDF KNN:** Nearest-neighbor search over address tokens to match entities with identical physical locations but disparate DBA ("Doing Business As") trade names.
4. **Phonetic Blocking:** Double Metaphone soundex encoding to cluster names that sound identical despite phonetic spelling differences.
5. **Sorted Neighborhood:** Windowed candidate sweep over standardized alphanumeric name prefixes.
* **Top 60 Candidate Selection:** Aggregates confidence scores across all five passes to isolate and retain the top 60 highest-probability candidates per reference entity.

### 3. Feature Engineering (`src/features.py`)
Constructed **25 numeric similarity features** for each candidate pair:
* **Edit & String Distance:** Normalized Levenshtein distance, Jaro-Winkler similarity, Longest Common Subsequence (LCS) ratio.
* **Set & Overlap Metrics:** Jaccard token similarity, Sørensen–Dice coefficient, token sort ratio, token set ratio.
* **Exact & Structural Matches:** Exact string match indicator, suffix-stripped name match, acronym match, string length absolute difference, string length ratio.
* **Address & Location Signals:** Postal PIN/ZIP code exact match indicator, street number match indicator, address token Jaccard overlap, country equality flag.

### 4. Classifier & Threshold Tuning (`src/train_model.py`, `src/decode.py`)
* **Model Configuration:** LightGBM Gradient Boosted Decision Tree (500 estimators, 31 max leaves, learning rate 0.05, feature fraction 0.8, binary logloss objective).
* **Threshold Optimization:** Grid search across probability thresholds ($0.05 \le \tau \le 0.95$ at 0.01 step intervals) directly evaluating the resulting macro $F_{0.5}$ score on validation sets.

---

## Repository Structure

```text
Amazon-ML-challenge-2026/
├── README.md                                  # Comprehensive architecture, metrics & execution guide
├── LICENSE                                    # MIT License
├── requirements.txt                           # Global project dependencies
├── Documentation_template.md                  # Completed methodology write-up for submission
├── amazon_ml_challenge_problem_statement.pdf  # Official challenge problem statement
├── submission.zip                             # Packaged submission archive
│
├── code/
│   └── business_entity_resolution/            # Official Submission Codebase
│       ├── README.md                          # Pipeline run instructions
│       ├── requirements.txt                   # Pinned pipeline dependencies
│       ├── output/                            # Trained models and pipeline artifacts
│       │   ├── lgbm_model.txt                 # Trained LightGBM model weights
│       │   ├── threshold.json                 # Optimal decision threshold
│       │   ├── metrics_summary.txt            # Training and validation metrics
│       │   └── val_predictions.csv            # Validation set predictions
│       └── src/                               # Modular Python source code
│           ├── __init__.py
│           ├── normalize.py                   # Text & address normalization module
│           ├── blocking.py                    # Multi-strategy candidate generation
│           ├── features.py                    # 25-feature string & address engineering
│           ├── train_model.py                 # LightGBM training & validation
│           ├── decode.py                      # Threshold optimization for F_0.5
│           ├── evaluate.py                    # Macro F_0.5 evaluation implementation
│           ├── write_outputs.py               # TSV generation & automated validation
│           ├── inspect_data.py                # Dataset inspection utilities
│           ├── matching.py                    # Core matching logic
│           ├── preprocessing.py               # Data loading & preparation
│           └── run_pipeline.py                # End-to-end orchestration entry point
│
├── output/                                    # Validated Submission Outputs
│   ├── matching_results.tsv                   # Final matched entity IDs (1,732,544 rows - PASS)
│   └── candidate_pairs.tsv                    # Final candidate entity IDs (1,732,544 rows - PASS)
│
├── notebooks/                                 # Exploratory Analysis & Experiments
│   ├── Amazon_ML_Challenge_2026_Entity_Resolution.ipynb  # Interactive 69-cell EDA & pipeline
│   ├── amazon_ml_challenge_pipeline.ipynb                # Streamlined chunked pipeline notebook
│   └── figures/                               # Visualizations and distribution plots
│
├── docs/                                      # Project Specifications & Notes
│   ├── facts.txt                              # Key impact points and resume highlights
│   ├── execution_prompt_specification.docx    # Autonomous execution specification
│   └── execution_prompt_specification.txt     # Plain-text execution specification
│
├── scripts/                                   # Standalone Scripts & Utilities
│   ├── check_download.py                      # Dataset verification utility
│   ├── fix_outputs.py                         # Output alignment and padding script
│   └── run_kaggle_solution.py                 # Standalone script for Kaggle/local execution
│
└── student_resource/                          # Official Dataset & Validation Tools
    ├── Documentation_template.md              # Blank submission template
    ├── README.md                              # Official problem description
    ├── dataset/                               # Train and test TSV files
    │   ├── train/                             # Ground truth & train sources 1, 2, 3
    │   └── test/                              # Test sources 1, 2, 3
    └── utils/
        └── validate_submission.py             # Official submission format validator
```

---

## Getting Started

### 1. Prerequisites
* Python 3.10 or higher
* Recommended: Dedicated virtual environment (`.venv` or conda)

### 2. Installation
```bash
git clone https://github.com/RaunakSachdeva2004/Amazon-ML-challenge-2026.git
cd Amazon-ML-challenge-2026
pip install -r requirements.txt
```

### 3. Running the Pipeline End-to-End
To run the complete data normalization, blocking, feature generation, model training, and submission output generation:
```bash
cd code/business_entity_resolution/src
python run_pipeline.py
```

Outputs will be automatically written to `code/business_entity_resolution/output/` and verified with the challenge validator.

### 4. Validating the Submission Format
Run the official challenge validator against the output directory:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
A valid submission produces:
```
PASS - no blocking issues found. Safe to submit.
```

---

## Submission Package Format

The submission archive `submission.zip` matches the official competition layout:
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # Final entity matches (1,732,544 rows)
│   └── candidate_pairs.tsv         # Blocking candidate set (1,732,544 rows)
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # Source code modules
│       ├── README.md               # Reproduction guide
│       └── requirements.txt        # Pinned dependencies
└── Documentation_template.md       # Completed methodology write-up
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.