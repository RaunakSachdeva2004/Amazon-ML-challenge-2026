# Amazon ML Challenge 2026: Multi-Source Business Entity Resolution

An end-to-end, high-performance machine learning pipeline for multi-source business entity resolution, developed for the Amazon ML Challenge 2026. The system links noisy, fragmented business records across three independent data sources without shared identifiers, optimizing for precision-weighted macro $F_{0.5}$ score.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |   Source 1 (Reference Entity Set)     |
                      |   Source 2 & Source 3 (Noisy Sources) |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Multi-Strategy Normalization     |
                      |  (Unicode NFKC, Suffixes, Addresses)  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        Multi-Strategy Blocking        |
                      |  - Token Inverted Index               |
                      |  - TF-IDF KNN (Name & Address)        |
                      |  - Phonetic Keying (Double Metaphone) |
                      |  - Sorted Neighborhood                |
                      +-------------------+-------------------+
                                          | (Top 60 Candidates / S1)
                                          v
                      +---------------------------------------+
                      |          Feature Engineering          |
                      |  - String Distance (Levenshtein,      |
                      |    Jaro-Winkler, Jaccard, Token Sort) |
                      |  - Address Overlap (ZIP, Numbers)     |
                      |  - Exact & Suffix-Stripped Matches    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       LightGBM Classifier Model       |
                      |       (Binary Objective, <=8B Params) |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Precision-Weighted Decoding      |
                      |      (Macro F_0.5 Threshold Tuning)   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |            Final Deliverables         |
                      |  - output/matching_results.tsv        |
                      |  - output/candidate_pairs.tsv         |
                      +---------------------------------------+
```

---

## Repository Structure

```text
Amazon-ML-challenge-2026/
├── README.md                                  # Repository overview and documentation
├── LICENSE                                    # MIT License
├── requirements.txt                           # Global project dependencies
├── Documentation_template.md                  # Filled methodology write-up for submission
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
│           ├── features.py                    # String similarity & address feature engineering
│           ├── train_model.py                 # LightGBM training & cross-validation
│           ├── decode.py                      # Threshold optimization for F_0.5
│           ├── evaluate.py                    # Macro F_0.5 evaluation implementation
│           ├── write_outputs.py               # TSV generation & automated validation
│           ├── inspect_data.py                # Dataset inspection utilities
│           ├── matching.py                    # Core matching logic
│           ├── preprocessing.py               # Data loading & preparation
│           └── run_pipeline.py                # End-to-end orchestration entry point
│
├── output/                                    # Validated Submission Outputs
│   ├── matching_results.tsv                   # Final matched entity IDs per Source-1 entity
│   └── candidate_pairs.tsv                    # Final candidate entity IDs per Source-1 entity
│
├── notebooks/                                 # Exploratory Analysis & Experiments
│   ├── Amazon_ML_Challenge_2026_Entity_Resolution.ipynb  # Interactive EDA & full pipeline
│   ├── amazon_ml_challenge_pipeline.ipynb                # Streamlined chunked pipeline notebook
│   └── figures/                               # Visualizations and distribution plots
│
├── docs/                                      # Project Specifications & Notes
│   ├── facts.txt                              # Key impact points and technical highlights
│   ├── execution_prompt_specification.docx    # Competition execution specification
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

## Key Pipeline Modules

1. **Text Normalization (`src/normalize.py`)**:
   - Cleans and standardizes noisy business entities across multiple countries (India, US, France).
   - Standardizes business suffixes (e.g., `corp`, `ltd`, `pvt`, `gmbh`, `sa`), address abbreviations (e.g., `rd`, `st`, `ave`), and strips special characters.
   - Extracts PIN/ZIP codes, street numbers, and structural address components.

2. **Candidate Generation / Blocking (`src/blocking.py`)**:
   - Reduces the quadratic comparison space $\mathcal{O}(N \times M)$ using 5 complementary blocking passes:
     1. Exact token inverted index
     2. Character n-gram TF-IDF KNN for business names
     3. Address token TF-IDF KNN
     4. Phonetic hashing (Double Metaphone)
     5. Sorted neighborhood on standardized name prefixes
   - Retains the top candidate pairs per Source-1 entity to ensure maximum recall while bounding memory.

3. **Feature Engineering (`src/features.py`)**:
   - Extracts fine-grained similarity signals:
     - Levenshtein distance, Jaro-Winkler, Jaccard token overlap, Dice coefficient
     - Token sort ratio, token set ratio
     - Substring containment and acronym matching
     - Address-specific matching: exact postal/PIN code match, street number match, city/state token overlap

4. **Model Training & Decoding (`src/train_model.py`, `src/decode.py`)**:
   - Trains a LightGBM gradient boosted decision tree classifier (under the 8-billion parameter and MIT license constraints).
   - Optimizes decision thresholds on held-out validation data directly against the competition metric: **macro $F_{0.5}$** across Source-1 entities.

---

## Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- Recommended: Dedicated virtual environment (`.venv` or conda)

### 2. Installation
```bash
git clone https://github.com/RaunakSachdeva2004/Amazon-ML-challenge-2026.git
cd Amazon-ML-challenge-2026
pip install -r requirements.txt
```

### 3. Running the Pipeline End-to-End
To run the full training, feature extraction, inference, and submission generation:
```bash
cd code/business_entity_resolution/src
python run_pipeline.py
```

Outputs will be automatically generated and verified.

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

The submission archive `submission.zip` matches the official challenge structure:
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