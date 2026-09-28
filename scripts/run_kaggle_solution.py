try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ============================================================
# 1. SETUP — FAST, LIGHTWEIGHT, KAGGLE-FRIENDLY
# ============================================================
# import os
import re
import gc
import time
import math
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Optional high-performance dataframe engine.
try:
    import polars as pl
    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

# Fast string similarity library. Kaggle commonly has it installed.
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

print("Environment ready")
print("Polars:", HAS_POLARS)
print("RapidFuzz:", HAS_RAPIDFUZZ)
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()
DATA_ROOT = PROJECT_ROOT / "student_resource"
print("Selected DATA_ROOT:", DATA_ROOT)


# List all files inside the directory
all_files = sorted(DATA_ROOT.rglob("*"))
for f in all_files:
    if f.is_file():
        print(" -", f.relative_to(DATA_ROOT))
# ============================================================
# 3. RESOLVE TRAIN / TEST PATHS
# ============================================================
def find_file(filename):
    hits = list(DATA_ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Could not find {filename}")
    return hits[0]

FILES = {
    "train_s1": find_file("train_source1.tsv"),
    "train_s2": find_file("train_source2.tsv"),
    "train_s3": find_file("train_source3.tsv"),
    "ground_truth": find_file("train_ground_truth.tsv"),
    "test_s1": find_file("test_source1.tsv"),
    "test_s2": find_file("test_source2.tsv"),
    "test_s3": find_file("test_source3.tsv"),
}

for k, v in FILES.items():
    print(f"{k:14s} -> {v}")
# ============================================================
# 4. FILE SIZES + SMALL PREVIEWS
# ============================================================
for k, path in FILES.items():
    size_mb = path.stat().st_size / (1024**2)
    print(f"{k:14s} {size_mb:10.1f} MB")

def preview_tsv(path, n=5):
    return pd.read_csv(path, sep="\t", nrows=n)

print("\nTRAIN SOURCE 1 PREVIEW")
print(preview_tsv(FILES["train_s1"]))

print("\nGROUND TRUTH PREVIEW")
print(preview_tsv(FILES["ground_truth"]))
# ============================================================
# 5. SCHEMA CHECK
# ============================================================
CORE_COLS = ["entity_id", "business_name", "business_address", "country"]

for name in ["train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"]:
    df = pd.read_csv(FILES[name], sep="\t", nrows=3)
    print(f"\n{name}")
    print(df.dtypes)
    print("columns:", list(df.columns))
# ============================================================
# 6. SAMPLE-BASED EDA
# ============================================================
SAMPLE_N = 50_000
RANDOM_STATE = 42

def sample_tsv(path, n=SAMPLE_N, seed=RANDOM_STATE):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=CORE_COLS,
        nrows=n
    )

eda_s1 = sample_tsv(FILES["train_s1"])
eda_s2 = sample_tsv(FILES["train_s2"])
eda_s3 = sample_tsv(FILES["train_s3"])

for label, df in [("S1", eda_s1), ("S2", eda_s2), ("S3", eda_s3)]:
    print(f"\n===== {label} =====")
    print("rows:", len(df))
    print("missing:")
    print(df.isna().mean().mul(100).round(2).to_frame("missing_%"))
    print("countries:")
    print(df["country"].value_counts(dropna=False).head(15).to_frame("count"))
# ============================================================
# 7. QUICK VISUAL DASHBOARD
# ============================================================




# ============================================================
# 8. NORMALIZATION FUNCTIONS
# ============================================================
LEGAL_SUFFIXES = {
    "incorporated", "inc", "corporation", "corp",
    "limited", "ltd", "llc", "llp",
    "private", "pvt", "company", "co",
    "limitedliabilitycompany"
}

def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = x.replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_name(x):
    s = normalize_text(x)
    tokens = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def normalize_address(x):
    return normalize_text(x)

def compact(s):
    return re.sub(r"\s+", "", s)

def token_signature(s):
    return " ".join(sorted(set(s.split())))

for df in [eda_s1]:
    df["norm_name"] = df["business_name"].map(normalize_name)
    df["norm_address"] = df["business_address"].map(normalize_address)
    df["compact_name"] = df["norm_name"].map(compact)
    df["compact_address"] = df["norm_address"].map(compact)

print(
    eda_s1[
        ["business_name", "norm_name", "business_address", "norm_address"]
    ].head(15)
)
# ============================================================
# 9. NORMALIZATION INSPECTION
# ============================================================
inspection = eda_s1[
    ["business_name", "norm_name", "business_address", "norm_address", "country"]
].copy()

print(inspection.sample(min(25, len(inspection)), random_state=42))
# ============================================================
# 10. GROUND-TRUTH SUMMARY
# ============================================================
gt = pd.read_csv(FILES["ground_truth"], sep="\t", dtype=str).fillna("")

def split_ids(x):
    if not x:
        return []
    return [v.strip() for v in x.split(",") if v.strip()]

gt["match_list"] = gt["matched_entity_ids"].map(split_ids)
gt["n_matches"] = gt["match_list"].str.len()

print("Ground-truth rows:", len(gt))
print("Singleton / no-match S1:", int((gt["n_matches"] == 0).sum()))
print("S1 with >=1 match:", int((gt["n_matches"] > 0).sum()))
print("Maximum matches for one S1:", int(gt["n_matches"].max()))

print(gt["n_matches"].describe().to_frame("match_count"))
# ============================================================
# 11. MATCH COUNT VISUALIZATION
# ============================================================
counts = gt["n_matches"].value_counts().sort_index()

if HAS_MATPLOTLIB:
    try:
        ax = counts.head(20).plot(kind="bar", figsize=(12, 4))
        ax.set_title("Ground Truth — Number of Matches per Source 1 Entity")
        ax.set_xlabel("Number of matched S2/S3 records")
        ax.set_ylabel("Number of S1 entities")
        plt.xticks(rotation=0)
    except Exception:
        pass


# ============================================================
# 12. EXACT MATCH HELPERS
# ============================================================
def prepare_small(df):
    out = df[CORE_COLS].copy()
    out = out.fillna("")
    out["norm_name"] = out["business_name"].map(normalize_name)
    out["norm_address"] = out["business_address"].map(normalize_address)
    out["compact_name"] = out["norm_name"].map(compact)
    out["compact_address"] = out["norm_address"].map(compact)
    out["name_addr_key"] = (
        out["country"].astype(str) + "|" +
        out["compact_name"] + "|" +
        out["compact_address"]
    )
    return out

small_s1 = prepare_small(pd.read_csv(FILES["train_s1"], sep="\t", nrows=20_000, dtype=str))
small_s2 = prepare_small(pd.read_csv(FILES["train_s2"], sep="\t", nrows=50_000, dtype=str))
small_s3 = prepare_small(pd.read_csv(FILES["train_s3"], sep="\t", nrows=50_000, dtype=str))

print(small_s1.head())
# ============================================================
# 13. BLOCKING INDEX
# ============================================================
def build_index(df, column):
    idx = defaultdict(list)
    for row in df.itertuples(index=False):
        key = getattr(row, column)
        if key:
            idx[key].append(row.entity_id)
    return idx

def add_block_keys(df):
    df = df.copy()
    df["country_name_key"] = (
        df["country"].astype(str) + "|" + df["compact_name"]
    )
    df["country_addr_key"] = (
        df["country"].astype(str) + "|" + df["compact_address"]
    )
    df["country_name_addr_key"] = (
        df["country"].astype(str) + "|" +
        df["compact_name"] + "|" + df["compact_address"]
    )
    return df

small_s1 = add_block_keys(small_s1)
small_s2 = add_block_keys(small_s2)
small_s3 = add_block_keys(small_s3)

s2_name_idx = build_index(small_s2, "country_name_key")
s2_addr_idx = build_index(small_s2, "country_addr_key")
s3_name_idx = build_index(small_s3, "country_name_key")
s3_addr_idx = build_index(small_s3, "country_addr_key")

print("S2 name index keys:", len(s2_name_idx))
print("S3 name index keys:", len(s3_name_idx))
# ============================================================
# 14. CANDIDATE GENERATION
# ============================================================
def candidates_for_row(row, indexes):
    candidates = set()

    for key_col, idx in [
        ("country_name_key", indexes["name"]),
        ("country_addr_key", indexes["addr"]),
        ("country_name_addr_key", indexes["name_addr"]),
    ]:
        key = getattr(row, key_col)
        if key:
            candidates.update(idx.get(key, []))

    return candidates

s2_name_addr_idx = build_index(small_s2, "country_name_addr_key")
s3_name_addr_idx = build_index(small_s3, "country_name_addr_key")

INDEXES_S2 = {
    "name": s2_name_idx,
    "addr": s2_addr_idx,
    "name_addr": s2_name_addr_idx,
}
INDEXES_S3 = {
    "name": s3_name_idx,
    "addr": s3_addr_idx,
    "name_addr": s3_name_addr_idx,
}

example = small_s1.iloc[0]
print("Example S1:", example["entity_id"])
print("S2 candidates:", candidates_for_row(example, INDEXES_S2))
print("S3 candidates:", candidates_for_row(example, INDEXES_S3))
# ============================================================
# 15. SCORING FUNCTIONS
# ============================================================
def jaccard_tokens(a, b):
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def pair_features(a, b):
    name_exact = int(a["compact_name"] != "" and a["compact_name"] == b["compact_name"])
    addr_exact = int(a["compact_address"] != "" and a["compact_address"] == b["compact_address"])
    name_token = jaccard_tokens(a["norm_name"], b["norm_name"])
    addr_token = jaccard_tokens(a["norm_address"], b["norm_address"])

    if HAS_RAPIDFUZZ:
        name_fuzzy = fuzz.token_set_ratio(a["norm_name"], b["norm_name"]) / 100
        addr_fuzzy = fuzz.token_set_ratio(a["norm_address"], b["norm_address"]) / 100
    else:
        name_fuzzy = float(name_exact)
        addr_fuzzy = float(addr_exact)

    return {
        "name_exact": name_exact,
        "addr_exact": addr_exact,
        "name_token": name_token,
        "addr_token": addr_token,
        "name_fuzzy": name_fuzzy,
        "addr_fuzzy": addr_fuzzy,
    }

def match_score(f):
    # Conservative weighting: exact agreement dominates.
    return (
        4.0 * f["name_exact"] +
        4.0 * f["addr_exact"] +
        2.0 * f["name_fuzzy"] +
        2.0 * f["addr_fuzzy"] +
        1.0 * f["name_token"] +
        1.0 * f["addr_token"]
    )
# ============================================================
# 16. INTERACTIVE PAIR INSPECTION
# ============================================================
def inspect_pair(a, b):
    f = pair_features(a, b)
    result = pd.DataFrame([{
        "S1": a["entity_id"],
        "candidate": b["entity_id"],
        "country": a["country"],
        "S1_name": a["business_name"],
        "candidate_name": b["business_name"],
        "S1_address": a["business_address"],
        "candidate_address": b["business_address"],
        **f,
        "score": match_score(f)
    }])
    print(result.T)

if len(small_s1) and len(small_s2):
    inspect_pair(small_s1.iloc[0], small_s2.iloc[0])
# ============================================================
# 17. F0.5 METRIC
# ============================================================
def f05_single(true_ids, pred_ids):
    true_set = set(true_ids)
    pred_set = set(pred_ids)

    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set) if true_set else 0.0

    if precision == 0 and recall == 0:
        return 0.0

    beta2 = 0.25
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)

def macro_f05(y_true, y_pred):
    return np.mean([
        f05_single(t, p)
        for t, p in zip(y_true, y_pred)
    ])
# ============================================================
# 18. VALIDATION SUBSET
# ============================================================
VAL_S1_N = 2_000

train_s1_val = pd.read_csv(
    FILES["train_s1"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str,
    nrows=VAL_S1_N
).fillna("")

train_s2_val = pd.read_csv(
    FILES["train_s2"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str
).fillna("")

train_s3_val = pd.read_csv(
    FILES["train_s3"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str
).fillna("")

train_s1_val = add_block_keys(prepare_small(train_s1_val))
train_s2_val = add_block_keys(prepare_small(train_s2_val))
train_s3_val = add_block_keys(prepare_small(train_s3_val))

print("Validation S1:", len(train_s1_val))
print("Validation S2:", len(train_s2_val))
print("Validation S3:", len(train_s3_val))
# ============================================================
# 19. VALIDATION INDEXES
# ============================================================
def make_indexes(df):
    return {
        "name": build_index(df, "country_name_key"),
        "addr": build_index(df, "country_addr_key"),
        "name_addr": build_index(df, "country_name_addr_key"),
    }

val_idx_s2 = make_indexes(train_s2_val)
val_idx_s3 = make_indexes(train_s3_val)

s2_by_id = train_s2_val.set_index("entity_id").to_dict("index")
s3_by_id = train_s3_val.set_index("entity_id").to_dict("index")
# ============================================================
# 20. CONSERVATIVE MATCH DECISION
# ============================================================
def is_match(f):
    # Highest-confidence rule
    if f["name_exact"] and f["addr_exact"]:
        return True

    # One exact field + strong support from the other
    if f["name_exact"] and f["addr_fuzzy"] >= 0.90:
        return True

    if f["addr_exact"] and f["name_fuzzy"] >= 0.92:
        return True

    # Both fields independently strong
    if f["name_fuzzy"] >= 0.96 and f["addr_fuzzy"] >= 0.88:
        return True

    return False

def predict_for_s1(row, idx2, idx3, by_id2, by_id3):
    candidates = set()
    candidates.update(candidates_for_row(row, idx2))
    candidates.update(candidates_for_row(row, idx3))

    predictions = []

    for cid in candidates:
        if cid.startswith("S2-"):
            c = by_id2.get(cid)
        else:
            c = by_id3.get(cid)

        if c is None:
            continue

        f = pair_features(row, c)
        if is_match(f):
            predictions.append(cid)

    return sorted(set(predictions))
# ============================================================
# 21. VALIDATION RUN
# ============================================================
gt_lookup = dict(zip(gt["source1_entity_id"], gt["match_list"]))

# Only score IDs that are present in this validation sample.
val_ids = set(train_s1_val["entity_id"])
val_gt = {k: v for k, v in gt_lookup.items() if k in val_ids}

predictions = {}
candidate_counts = []

t0 = time.time()

# Change this line in Section 21:
for _, row in train_s1_val.iterrows():
    pred = predict_for_s1(
        row, val_idx_s2, val_idx_s3, s2_by_id, s3_by_id
    )
    predictions[row["entity_id"]] = pred

elapsed = time.time() - t0

eval_ids = list(val_gt.keys())
y_true = [val_gt[k] for k in eval_ids]
y_pred = [predictions.get(k, []) for k in eval_ids]

score = macro_f05(y_true, y_pred)

print(f"Validation entities: {len(eval_ids):,}")
print(f"Validation time: {elapsed:.2f}s")
print(f"Macro F0.5: {score:.5f}")
print(f"Predicted links: {sum(map(len, y_pred)):,}")
print(f"Predicted singletons: {sum(len(x)==0 for x in y_pred):,}")
# ============================================================
# 22. ERROR ANALYSIS TABLE
# ============================================================
rows = []

for sid in eval_ids:
    t = set(val_gt[sid])
    p = set(predictions.get(sid, []))

    rows.append({
        "source1_entity_id": sid,
        "true_count": len(t),
        "pred_count": len(p),
        "TP": len(t & p),
        "FP": len(p - t),
        "FN": len(t - p),
        "F0.5": f05_single(t, p)
    })

errors = pd.DataFrame(rows)

print(errors.sort_values(["FP", "FN"], ascending=False).head(25))
print("\nAggregate:")
print(errors[["TP", "FP", "FN", "F0.5"]].sum().to_frame("sum"))
# ============================================================
# 23. SIMPLE THRESHOLD EXPERIMENT
# ============================================================
def make_rule(name_threshold, addr_threshold):
    def rule(f):
        if f["name_exact"] and f["addr_exact"]:
            return True
        if f["name_exact"] and f["addr_fuzzy"] >= addr_threshold:
            return True
        if f["addr_exact"] and f["name_fuzzy"] >= name_threshold:
            return True
        if f["name_fuzzy"] >= name_threshold and f["addr_fuzzy"] >= addr_threshold:
            return True
        return False
    return rule

threshold_results = []

# Small grid — intentionally cheap.
for nt in [0.92, 0.94, 0.96]:
    for at in [0.84, 0.88, 0.92]:
        rule = make_rule(nt, at)
        preds = {}

        for row in train_s1_val.itertuples(index=False):
            cand = set()
            cand.update(candidates_for_row(row, val_idx_s2))
            cand.update(candidates_for_row(row, val_idx_s3))

            # Convert row namedtuple to dict for pair_features compatibility
            row_dict = row._asdict() if hasattr(row, "_asdict") else row

            out = []
            for cid in cand:
                c = s2_by_id.get(cid) if cid.startswith("S2-") else s3_by_id.get(cid)
                if c is None:
                    continue
                
                # Convert candidate namedtuple/object to dict if needed
                c_dict = c._asdict() if hasattr(c, "_asdict") else c

                if rule(pair_features(row_dict, c_dict)):
                    out.append(cid)
            preds[row.entity_id] = sorted(set(out))

        yp = [preds.get(k, []) for k in eval_ids]
        threshold_results.append({
            "name_threshold": nt,
            "address_threshold": at,
            "macro_F0.5": macro_f05(y_true, yp),
            "predicted_links": sum(map(len, yp))
        })

threshold_df = pd.DataFrame(threshold_results).sort_values(
    "macro_F0.5", ascending=False
)
print(threshold_df)
# ============================================================
# 24. SCALABLE DATA LOADER
# ============================================================
def load_core(path, limit=None):
    kwargs = {
        "sep": "\t",
        "usecols": CORE_COLS,
        "dtype": str,
    }
    if limit is not None:
        kwargs["nrows"] = limit

    df = pd.read_csv(path, **kwargs).fillna("")
    return add_block_keys(prepare_small(df))

# Development mode:
FAST_DEV_MODE = True

DEV_S1_LIMIT = 25_000
DEV_S2_LIMIT = 100_000
DEV_S3_LIMIT = 100_000

if FAST_DEV_MODE:
    test_s1 = load_core(FILES["test_s1"], DEV_S1_LIMIT)
    test_s2 = load_core(FILES["test_s2"], DEV_S2_LIMIT)
    test_s3 = load_core(FILES["test_s3"], DEV_S3_LIMIT)
else:
    test_s1 = load_core(FILES["test_s1"])
    test_s2 = load_core(FILES["test_s2"])
    test_s3 = load_core(FILES["test_s3"])

print("Test S1:", len(test_s1))
print("Test S2:", len(test_s2))
print("Test S3:", len(test_s3))
# ============================================================
# 25. FINAL TEST INDEXES
# ============================================================
test_idx_s2 = make_indexes(test_s2)
test_idx_s3 = make_indexes(test_s3)

test_s2_by_id = test_s2.set_index("entity_id").to_dict("index")
test_s3_by_id = test_s3.set_index("entity_id").to_dict("index")

print("S2 name blocks:", len(test_idx_s2["name"]))
print("S3 name blocks:", len(test_idx_s3["name"]))
# ============================================================
# 26. CANDIDATE GENERATION
# ============================================================
def candidate_ids_for_test_row(row):
    out = set()
    out.update(candidates_for_row(row, test_idx_s2))
    out.update(candidates_for_row(row, test_idx_s3))
    return sorted(out)

candidate_rows = []
t0 = time.time()

for row in test_s1.itertuples(index=False):
    cands = candidate_ids_for_test_row(row)
    candidate_rows.append({
        "source1_entity_id": row.entity_id,
        "candidate_entity_ids": ",".join(cands)
    })

candidate_df = pd.DataFrame(candidate_rows)

print("Candidate generation time:", round(time.time() - t0, 2), "seconds")
print("S1 entities:", len(candidate_df))
print("Average candidates:", candidate_df["candidate_entity_ids"].map(
    lambda x: len(x.split(",")) if x else 0
).mean())
print(candidate_df.head(10))
# ============================================================
# 27. FINAL MATCHING
# ============================================================
def final_predictions_for_row(row):
    cands = candidate_ids_for_test_row(row)
    out = []

    # Convert row namedtuple to dict for pair_features compatibility
    row_dict = row._asdict() if hasattr(row, "_asdict") else row

    for cid in cands:
        if cid.startswith("S2-"):
            c = test_s2_by_id.get(cid)
        elif cid.startswith("S3-"):
            c = test_s3_by_id.get(cid)
        else:
            c = None

        if c is None:
            continue

        # Convert candidate namedtuple/object to dict if needed
        c_dict = c._asdict() if hasattr(c, "_asdict") else c

        f = pair_features(row_dict, c_dict)
        if is_match(f):
            out.append(cid)

    return sorted(set(out))

match_rows = []
t0 = time.time()

for row in test_s1.itertuples(index=False):
    matches = final_predictions_for_row(row)
    match_rows.append({
        "source1_entity_id": row.entity_id,
        "matched_entity_ids": ",".join(matches)
    })

matching_df = pd.DataFrame(match_rows)

print("Matching time:", round(time.time() - t0, 2), "seconds")
print(matching_df.head(10))
# ============================================================
# 28. NOTEBOOK VALIDATION
# ============================================================
def validate_outputs(s1_df, matching_df, candidate_df, s2_df, s3_df):
    problems = []

    expected = set(s1_df["entity_id"])
    predicted = set(matching_df["source1_entity_id"])

    if expected != predicted:
        problems.append(
            f"S1 coverage mismatch: expected {len(expected)}, got {len(predicted)}"
        )

    if matching_df["source1_entity_id"].duplicated().any():
        problems.append("Duplicate source1_entity_id in matching output.")

    valid_s2s3 = set(s2_df["entity_id"]) | set(s3_df["entity_id"])

    candidate_map = dict(zip(
        candidate_df["source1_entity_id"],
        candidate_df["candidate_entity_ids"].map(
            lambda x: set(x.split(",")) if x else set()
        )
    ))

    for row in matching_df.itertuples(index=False):
        mids = set(row.matched_entity_ids.split(",")) if row.matched_entity_ids else set()

        if len(mids) != len(row.matched_entity_ids.split(",")) if row.matched_entity_ids else False:
            problems.append(f"Duplicate IDs for {row.source1_entity_id}")

        bad_ids = mids - valid_s2s3
        if bad_ids:
            problems.append(f"Invalid IDs for {row.source1_entity_id}: {list(bad_ids)[:3]}")

        if not mids.issubset(candidate_map.get(row.source1_entity_id, set())):
            problems.append(f"Match outside candidate set: {row.source1_entity_id}")

    if problems:
        print("❌ Validation found issues:")
        for p in problems[:25]:
            print("-", p)
        return False

    print("✅ Notebook validation passed.")
    return True

validate_outputs(
    test_s1,
    matching_df,
    candidate_df,
    test_s2,
    test_s3
)
# ============================================================
# 29. SAVE OUTPUTS
# ============================================================
if Path("/kaggle/working").exists():
    OUTPUT_DIR = Path("/kaggle/working/output")
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matching_path = OUTPUT_DIR / "matching_results.tsv"
candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

matching_df.to_csv(matching_path, sep="\t", index=False)
candidate_df.to_csv(candidate_path, sep="\t", index=False)

print("Saved:")
print(matching_path)
print(candidate_path)

print("\nFile sizes:")
print("matching_results.tsv:", round(matching_path.stat().st_size / 1024**2, 2), "MB")
print("candidate_pairs.tsv:", round(candidate_path.stat().st_size / 1024**2, 2), "MB")
# ============================================================
# 30. FINAL DIAGNOSTICS
# ============================================================
match_counts = matching_df["matched_entity_ids"].map(
    lambda x: len(x.split(",")) if x else 0
)

candidate_counts = candidate_df["candidate_entity_ids"].map(
    lambda x: len(x.split(",")) if x else 0
)

diagnostics = pd.Series({
    "S1 entities": len(matching_df),
    "Predicted links": int(match_counts.sum()),
    "Predicted singleton S1": int((match_counts == 0).sum()),
    "Singleton rate %": round((match_counts == 0).mean() * 100, 2),
    "Average matches / S1": round(match_counts.mean(), 4),
    "Maximum matches / S1": int(match_counts.max()),
    "Average candidates / S1": round(candidate_counts.mean(), 4),
    "Maximum candidates / S1": int(candidate_counts.max()),
})

print(diagnostics.to_frame("value"))
# ============================================================
# 31. MULTI-MATCH AUDIT
# ============================================================
audit = matching_df.copy()
audit["match_count"] = match_counts

print(
    audit.sort_values("match_count", ascending=False)
         .head(30)
)
# ============================================================
# 35. FINAL QUICK CHECK
# ============================================================
print("=" * 70)
print("AMAZON ML CHALLENGE 2026 — FINAL CHECK")
print("=" * 70)
print("matching_results.tsv :", matching_path.exists())
print("candidate_pairs.tsv :", candidate_path.exists())
print("S1 rows              :", len(matching_df))
print("Predicted links      :", int(match_counts.sum()))
print("Predicted singletons :", int((match_counts == 0).sum()))
print("Avg candidates       :", round(candidate_counts.mean(), 3))
print("=" * 70)

# Helpful download links in the Kaggle output panel:
print(f"Output directory: {OUTPUT_DIR}")
