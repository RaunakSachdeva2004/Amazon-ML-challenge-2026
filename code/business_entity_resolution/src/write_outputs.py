import pandas as pd
import subprocess
import sys
import os
from collections import defaultdict

def write_test_outputs(test_s1_ids, df_test_cands, test_preds_dict, output_dir='../output', test_dir='../../../student_resource/dataset/test'):
    append_test_outputs(test_s1_ids, df_test_cands, test_preds_dict, output_dir, write_header=True, mode='w')
    run_validation(output_dir, test_dir)

def append_test_outputs(test_s1_ids, df_test_cands, test_preds_dict, output_dir='../output', write_header=False, mode='a'):
    # 1. matching_results.tsv
    matching_rows = []
    for s1 in test_s1_ids:
        matches = test_preds_dict.get(s1, [])
        matches = list(dict.fromkeys(matches))
        matches_str = ",".join(matches) if matches else ""
        matching_rows.append({'source1_entity_id': s1, 'matched_entity_ids': matches_str})
        
    df_matching = pd.DataFrame(matching_rows)
    matching_path = f'{output_dir}/matching_results.tsv'
    df_matching.to_csv(matching_path, sep='\t', index=False, mode=mode, header=write_header)
    
    # 2. candidate_pairs.tsv
    cand_dict = defaultdict(list)
    for idx, row in df_test_cands.iterrows():
        cand_dict[row['source1_entity_id']].append(row['candidate_entity_id'])
        
    cand_rows = []
    for s1 in test_s1_ids:
        cands = cand_dict.get(s1, [])
        cands = list(dict.fromkeys(cands))
        cands_str = ",".join(cands) if cands else ""
        cand_rows.append({'source1_entity_id': s1, 'candidate_entity_ids': cands_str})
        
    df_candidate = pd.DataFrame(cand_rows)
    candidate_path = f'{output_dir}/candidate_pairs.tsv'
    df_candidate.to_csv(candidate_path, sep='\t', index=False, mode=mode, header=write_header)

def run_validation(output_dir='../output', test_dir='../../../student_resource/dataset/test'):
    print("Running validation script...")
    matching_path = f'{output_dir}/matching_results.tsv'
    candidate_path = f'{output_dir}/candidate_pairs.tsv'
    val_script = '../../../student_resource/utils/validate_submission.py'
    cmd = [sys.executable, val_script, '--matching', matching_path, '--candidate', candidate_path, '--test-dir', test_dir]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Validation Output:")
    print(result.stdout)
    if result.returncode != 0:
        print("Validation Errors:")
        print(result.stderr)
        raise RuntimeError("Validation failed. See output above.")
    else:
        print("Validation PASS.")
