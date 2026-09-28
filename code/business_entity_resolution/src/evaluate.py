def f0_5_per_entity(predicted_ids: set, true_ids: set) -> float:
    if len(true_ids) == 0:
        return 1.0 if len(predicted_ids) == 0 else 0.0
    if len(predicted_ids) == 0:
        return 0.0
    tp = len(predicted_ids & true_ids)
    precision = tp / len(predicted_ids)
    recall = tp / len(true_ids)
    if precision == 0 and recall == 0:
        return 0.0
    beta = 0.5
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

def macro_f0_5(predictions: dict, ground_truth: dict) -> float:
    scores = [f0_5_per_entity(predictions.get(sid, set()), true_ids)
              for sid, true_ids in ground_truth.items()]
    if not scores: return 0.0
    return sum(scores) / len(scores)

def blocking_recall(candidates_df, ground_truth_dict):
    # candidates_df has source1_entity_id, candidate_entity_id
    grouped = candidates_df.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
    
    total_true_matches = 0
    found_true_matches = 0
    
    for sid, true_ids in ground_truth_dict.items():
        if not true_ids:
            continue
        cands = grouped.get(sid, set())
        total_true_matches += len(true_ids)
        found_true_matches += len(true_ids & cands)
        
    return found_true_matches / total_true_matches if total_true_matches > 0 else 1.0
