# Business Entity Resolution Pipeline

## Instructions to Run End-to-End

### 1. Requirements & Setup
Install required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Pipeline
Execute the main pipeline script from the root directory:
```bash
python code/business_entity_resolution/src/pipeline.py
```

This will automatically:
1. Load `dataset/test/test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`.
2. Apply text and address preprocessing & standardization routines.
3. Perform TF-IDF and n-gram shingle blocking to generate candidate pairs in `output/candidate_pairs.tsv`.
4. Extract pairwise similarity features (string distances, Jaccard overlaps, digit matching).
5. Predict match probabilities using trained Gradient Boosted decision trees.
6. Generate final leaderboard prediction file `output/matching_results.tsv`.
