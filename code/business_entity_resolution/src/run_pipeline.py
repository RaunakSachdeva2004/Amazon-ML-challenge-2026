import pandas as pd
import numpy as np
import time
import os
import gc
from normalize import process_dataframe
from blocking import run_blocking
from features import extract_features
from train_model import train_and_evaluate, construct_labels
from decode import tune_threshold, apply_threshold
from evaluate import blocking_recall
from write_outputs import append_test_outputs, run_validation
import json

def load_data(path, nrows=None):
    if nrows:
        return pd.read_csv(path, sep='\t', dtype=str, nrows=nrows).fillna("")
    return pd.read_csv(path, sep='\t', dtype=str).fillna("")

def smart_load_train_s23(gt_df, s1_train, s2_path, s3_path, bg_size=60000):
    s1_ids = set(s1_train['entity_id'])
    gt_df = gt_df[gt_df['source1_entity_id'].isin(s1_ids)]
    targets = set()
    for m in gt_df['matched_entity_ids'].dropna():
        if m != 'nan':
            targets.update(m.split(','))
    
    s2_targets = {t for t in targets if t.startswith('S2-')}
    s3_targets = {t for t in targets if t.startswith('S3-')}
    
    def fetch_with_targets(path, tgt_set):
        chunks = [pd.read_csv(path, sep='\t', dtype=str, nrows=bg_size).fillna("")]
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=250000):
            chunk = chunk.fillna("")
            matched = chunk[chunk['entity_id'].isin(tgt_set)]
            if not matched.empty:
                chunks.append(matched)
                tgt_set.difference_update(set(matched['entity_id']))
                if not tgt_set:
                    break
        return pd.concat(chunks).drop_duplicates(subset=['entity_id']).reset_index(drop=True)
        
    s2 = fetch_with_targets(s2_path, s2_targets)
    s3 = fetch_with_targets(s3_path, s3_targets)
    return s2, s3

def main():
    start_time = time.time()
    
    # 1. Load Train Data
    print("Loading train data...")
    df_train_s1 = load_data('../../../student_resource/dataset/train/train_source1.tsv')
    df_train_gt = pd.read_csv('../../../student_resource/dataset/train/train_ground_truth.tsv', sep='\t', dtype=str)
    
    df_train_s2, df_train_s3 = smart_load_train_s23(
        df_train_gt, df_train_s1, 
        '../../../student_resource/dataset/train/train_source2.tsv',
        '../../../student_resource/dataset/train/train_source3.tsv',
        bg_size=60000
    )
    
    # Process ground truth into dict
    gt_dict = {}
    for idx, row in df_train_gt.iterrows():
        s1 = row['source1_entity_id']
        matches = str(row['matched_entity_ids'])
        if pd.isna(row['matched_entity_ids']) or matches == 'nan' or not matches:
            gt_dict[s1] = set()
        else:
            gt_dict[s1] = set(matches.split(','))
            
    # 2. Normalize
    print("Normalizing train data...")
    df_train_s1 = process_dataframe(df_train_s1)
    df_train_s2 = process_dataframe(df_train_s2)
    df_train_s3 = process_dataframe(df_train_s3)
    df_train_s23 = pd.concat([df_train_s2, df_train_s3], ignore_index=True)
    del df_train_s2, df_train_s3
    gc.collect()
    
    # 3. Train / Val Split (80/20)
    print("Creating Train/Val Split...")
    np.random.seed(42)
    s1_ids = df_train_s1['entity_id'].values.copy()
    np.random.shuffle(s1_ids)
    split_idx = int(len(s1_ids) * 0.8)
    train_ids = s1_ids[:split_idx]
    val_ids = s1_ids[split_idx:]
    
    df_train_split = df_train_s1[df_train_s1['entity_id'].isin(train_ids)].copy()
    df_val_split = df_train_s1[df_train_s1['entity_id'].isin(val_ids)].copy()
    
    # 4. Blocking on Train Split
    print("Running blocking on Train split...")
    df_cands_train, train_indices = run_blocking(df_train_split, df_train_s23, top_k=20)
    
    # 5. Blocking on Val Split
    print("Running blocking on Val split...")
    df_cands_val, _ = run_blocking(df_val_split, df_train_s23, top_k=20, indices=train_indices)
    
    # Measure blocking recall
    val_recall = blocking_recall(df_cands_val, {k: gt_dict[k] for k in val_ids})
    print(f"Validation Blocking Recall (top 20): {val_recall:.4f}")
    
    # 6. Feature Extraction
    print("Extracting features for Train split...")
    X_train = extract_features(df_train_split, df_train_s23, df_cands_train)
    y_train = construct_labels(df_cands_train, gt_dict)
    
    print("Extracting features for Val split...")
    X_val = extract_features(df_val_split, df_train_s23, df_cands_val)
    y_val = construct_labels(df_cands_val, gt_dict)
    
    # 7. Train Model
    model = train_and_evaluate(X_train, X_val, y_train, y_val)
    
    # 8. Decode / Threshold
    df_val_res = pd.read_csv('../output/val_predictions.csv')
    best_thresh, best_f05 = tune_threshold(df_val_res, {k: gt_dict[k] for k in val_ids})
    
    # Free memory
    del df_train_s1, df_train_s23, train_indices
    del df_train_split, df_val_split, df_cands_train, df_cands_val, X_train, X_val, y_train, y_val
    gc.collect()
    
    # 9. Test Set Evaluation (Chunked)
    print("\n--- Starting Chunked Test Set Evaluation ---")
    print("Loading Test Target Data (S2, S3)...")
    df_test_s2 = load_data('../../../student_resource/dataset/test/test_source2.tsv')
    df_test_s3 = load_data('../../../student_resource/dataset/test/test_source3.tsv')
    
    print("Normalizing test targets...")
    df_test_s2 = process_dataframe(df_test_s2)
    df_test_s3 = process_dataframe(df_test_s3)
    df_test_s23 = pd.concat([df_test_s2, df_test_s3], ignore_index=True)
    del df_test_s2, df_test_s3
    gc.collect()
    
    print("Building Test Blocking Indices...")
    _, test_indices = run_blocking(df_test_s23.head(1), df_test_s23, top_k=20)
    
    print("Processing Test Source 1 in chunks...")
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_size = 25000
    chunk_iter = pd.read_csv('../../../student_resource/dataset/test/test_source1.tsv', sep='\t', dtype=str, chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iter):
        print(f"\n[Chunk {i+1}] Processing {len(chunk)} rows...")
        chunk = chunk.fillna("")
        chunk = process_dataframe(chunk)
        
        df_cands_test, _ = run_blocking(chunk, df_test_s23, top_k=20, indices=test_indices)
        X_test = extract_features(chunk, df_test_s23, df_cands_test)
        
        drop_cols = ['source1_entity_id', 'candidate_entity_id']
        X_test_features = X_test.drop(columns=drop_cols)
        test_preds = model.predict_proba(X_test_features)[:, 1]
        
        df_test_res = df_cands_test[['source1_entity_id', 'candidate_entity_id']].copy()
        df_test_res['score'] = test_preds
        
        test_preds_dict = apply_threshold(df_test_res, best_thresh)
        
        # Write mode: 'w' for the first chunk to overwrite old files, 'a' to append
        mode = 'w' if i == 0 else 'a'
        write_header = True if i == 0 else False
        append_test_outputs(chunk['entity_id'].tolist(), df_cands_test, test_preds_dict, output_dir=output_dir, write_header=write_header, mode=mode)
        
        del chunk, df_cands_test, X_test, X_test_features, df_test_res
        gc.collect()
        
    print("\nAll chunks processed! Validating final output files...")
    run_validation(output_dir='../output', test_dir='../../../student_resource/dataset/test')
    
    print(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")

if __name__ == '__main__':
    main()
