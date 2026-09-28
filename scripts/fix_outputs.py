import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"
OUT_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "output"
VAL_SCRIPT = PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"

print("Loading test_source1...")
test_s1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", usecols=["entity_id"])

print("Padding matching_results.tsv...")
matching_path = OUT_DIR / "matching_results.tsv"
matching = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
matching_dict = dict(zip(matching["source1_entity_id"], matching["matched_entity_ids"]))

all_matching_rows = [{"source1_entity_id": eid, "matched_entity_ids": matching_dict.get(eid, "")} for eid in test_s1["entity_id"]]
pd.DataFrame(all_matching_rows).to_csv(matching_path, sep="\t", index=False)

print("Padding candidate_pairs.tsv...")
cands_path = OUT_DIR / "candidate_pairs.tsv"
cands = pd.read_csv(cands_path, sep="\t", dtype=str).fillna("")
cands_dict = dict(zip(cands["source1_entity_id"], cands["candidate_entity_ids"]))

all_cands_rows = [{"source1_entity_id": eid, "candidate_entity_ids": cands_dict.get(eid, "")} for eid in test_s1["entity_id"]]
pd.DataFrame(all_cands_rows).to_csv(cands_path, sep="\t", index=False)

print("Done padding. Running validation...")
cmd = [
    sys.executable,
    str(VAL_SCRIPT),
    "--matching", str(matching_path),
    "--candidate", str(cands_path),
    "--test-dir", str(TEST_DIR)
]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr)
