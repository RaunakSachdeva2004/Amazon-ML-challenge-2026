"""
ML Matching Model trainer & predictor using LightGBM / XGBoost.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Set, List, Tuple
from .evaluate import compute_entity_f_score

class MatchingModel:
    def __init__(self, probability_threshold: float = 0.65):
        self.probability_threshold = probability_threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Fits the binary LightGBM classifier."""
        self.model.fit(X, y)

    def tune_threshold(
        self, 
        candidate_df: pd.DataFrame, 
        probs: np.ndarray, 
        gt_map: Dict[str, Set[str]],
        beta: float = 0.5
    ) -> float:
        """Finds probability threshold that maximizes macro-averaged F_0.5 score."""
        candidate_df = candidate_df.copy()
        candidate_df['prob'] = probs

        best_thresh = 0.5
        best_score = -1.0

        thresholds = np.linspace(0.40, 0.90, 51)
        for th in thresholds:
            preds_map: Dict[str, Set[str]] = {s1_id: set() for s1_id in gt_map.keys()}
            filtered = candidate_df[candidate_df['prob'] >= th]
            
            for _, row in filtered.iterrows():
                preds_map[row['s1_id']].add(row['cand_id'])

            scores = [compute_entity_f_score(preds_map.get(s1, set()), true_ids, beta=beta) 
                      for s1, true_ids in gt_map.items()]
            macro_f = float(np.mean(scores))
            
            if macro_f > best_score:
                best_score = macro_f
                best_thresh = th

        self.probability_threshold = float(best_thresh)
        print(f"Optimal probability threshold: {self.probability_threshold:.3f} (Macro F_{beta}: {best_score:.4f})")
        return self.probability_threshold

    def predict(self, candidate_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
        """
        Formats final matching_results.tsv DataFrame from candidate probabilities.
        """
        candidate_df = candidate_df.copy()
        candidate_df['prob'] = probs
        
        filtered = candidate_df[candidate_df['prob'] >= self.probability_threshold]

        # Group matched candidate entity IDs per Source 1 entity
        grouped = filtered.groupby('s1_id')['cand_id'].apply(lambda x: ",".join(sorted(set(x)))).to_dict()

        all_s1_ids = candidate_df['s1_id'].unique()
        rows = []
        for s1_id in all_s1_ids:
            matched_str = grouped.get(s1_id, "")
            rows.append({'source1_entity_id': s1_id, 'matched_entity_ids': matched_str})

        return pd.DataFrame(rows)
