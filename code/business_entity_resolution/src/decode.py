import pandas as pd
import numpy as np
from evaluate import macro_f0_5
import json
from collections import defaultdict

def tune_threshold(df_val_res, ground_truth_dict):
    print("Tuning threshold...")
    # df_val_res has source1_entity_id, candidate_entity_id, score, label
    
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thresh = 0.5
    best_f05 = 0.0
    
    for thresh in thresholds:
        preds_dict = defaultdict(set)
        # keep only candidates >= thresh
        accepted = df_val_res[df_val_res['score'] >= thresh]
        
        for idx, row in accepted.iterrows():
            preds_dict[row['source1_entity_id']].add(row['candidate_entity_id'])
            
        f05 = macro_f0_5(preds_dict, ground_truth_dict)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"Best threshold: {best_thresh:.2f}, Validation F0.5: {best_f05:.4f}")
    
    with open('../output/threshold.json', 'w') as f:
        json.dump({'best_threshold': best_thresh}, f)
        
    return best_thresh, best_f05

def apply_threshold(df_res, threshold):
    preds_dict = defaultdict(list)
    accepted = df_res[df_res['score'] >= threshold]
    
    for idx, row in accepted.iterrows():
        preds_dict[row['source1_entity_id']].append(row['candidate_entity_id'])
        
    return preds_dict
