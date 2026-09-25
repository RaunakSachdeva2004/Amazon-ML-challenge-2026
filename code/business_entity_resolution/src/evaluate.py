"""
Evaluation module for macro-averaged F_0.5 metric calculation.
"""
import pandas as pd
import numpy as np

def compute_entity_f_score(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
    """
    Calculates F_beta score for a single Source 1 entity.
    Singletons:
    - If true_ids is empty and pred_ids is empty -> returns 1.0
    - If true_ids is empty and pred_ids is non-empty -> returns 0.0
    """
    if len(true_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0
    if len(pred_ids) == 0:
        return 0.0

    tp = len(pred_ids.intersection(true_ids))
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    beta_sq = beta ** 2
    f_beta = ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)
    return f_beta

def evaluate_predictions(ground_truth_df: pd.DataFrame, predictions_df: pd.DataFrame, beta: float = 0.5) -> float:
    """
    Calculates the macro-averaged F_beta score over all Source 1 entities.
    Expects DataFrames with columns ['source1_entity_id', 'matched_entity_ids'].
    """
    # Create lookup dicts
    gt_map = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        raw_ids = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        ids = set(x.strip() for x in raw_ids.split(',') if x.strip())
        gt_map[s1_id] = ids

    pred_map = {}
    for _, row in predictions_df.iterrows():
        s1_id = row['source1_entity_id']
        raw_ids = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        ids = set(x.strip() for x in raw_ids.split(',') if x.strip())
        pred_map[s1_id] = ids

    scores = []
    for s1_id, true_ids in gt_map.items():
        p_ids = pred_map.get(s1_id, set())
        score = compute_entity_f_score(p_ids, true_ids, beta=beta)
        scores.append(score)

    return float(np.mean(scores))
