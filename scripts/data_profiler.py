# scripts/data_profiler.py
# MIRA v1.0 — Universal Data Exploration Script
# Works on ANY binary classification dataset
# Dataset-agnostic: no hardcoded column names

import pandas as pd
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"
TARGET_COL   = sys.argv[2] if len(sys.argv) > 2 else "Loan Status"
OUTPUT_PATH  = sys.argv[3] if len(sys.argv) > 3 else "processed/data_exploration.json"

print(f"\n{'='*55}")
print(f"  MIRA v1.0 — Data Exploration")
print(f"{'='*55}")
print(f"  Dataset : {DATASET_PATH}")
print(f"  Target  : {TARGET_COL}")
print(f"  Output  : {OUTPUT_PATH}")
print(f"{'='*55}\n")

# ── Load dataset ───────────────────────────────────────────────
print("📂 Loading dataset...")
if DATASET_PATH.endswith(".csv"):
    df = pd.read_csv(DATASET_PATH)
elif DATASET_PATH.endswith((".xlsx", ".xls")):
    df = pd.read_excel(DATASET_PATH)
elif DATASET_PATH.endswith(".parquet"):
    df = pd.read_parquet(DATASET_PATH)
else:
    df = pd.read_csv(DATASET_PATH)

print(f"   Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Validate target ────────────────────────────────────────────
if TARGET_COL not in df.columns:
    print(f"\n❌ Target '{TARGET_COL}' not found!")
    print(f"   Available: {list(df.columns)}")
    sys.exit(1)

# ── Basic stats ────────────────────────────────────────────────
row_count   = int(df.shape[0])
col_count   = int(df.shape[1])
missing_pct = float(round(df.isnull().mean().mean() * 100, 4))
dup_count   = int(df.duplicated().sum())

print(f"   Missing : {missing_pct}%")
print(f"   Dupes   : {dup_count}")

# ── Infer task type ────────────────────────────────────────────
target_unique = df[TARGET_COL].nunique()
target_dtype  = str(df[TARGET_COL].dtype)

if target_unique <= 10 or target_dtype in ['object', 'bool', 'str']:
    task_type = "classification"
else:
    task_type = "regression"
print(f"   Task    : {task_type} ({target_unique} unique values)")

# ── Target distribution ────────────────────────────────────────
target_counts  = df[TARGET_COL].value_counts()
target_dist    = {str(k): int(v) for k, v in target_counts.items()}
minority       = int(target_counts.min())
total          = int(target_counts.sum())
imbalance_ratio = round(float(minority / total), 4)
imbalance      = bool(imbalance_ratio < 0.20)

print(f"   Distribution: {target_dist}")
print(f"   Imbalance   : {imbalance} (ratio: {imbalance_ratio})")

# ── Column classification ──────────────────────────────────────
# ID columns — unique value per row
id_cols = [
    col for col in df.columns
    if col != TARGET_COL
    and df[col].nunique() == len(df)
]

# Constant columns
const_cols = [
    col for col in df.columns
    if col != TARGET_COL
    and df[col].nunique() <= 1
]

# Numeric columns (excluding ID and constant)
num_cols = [
    col for col in df.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    if col not in id_cols
    and col not in const_cols
]

# Categorical columns
cat_cols = [
    col for col in df.select_dtypes(
        exclude=['int64', 'float64']
    ).columns.tolist()
    if col not in id_cols
    and col not in const_cols
]

# High cardinality text columns
text_cols = [
    col for col in cat_cols
    if df[col].nunique() > 50
]

print(f"\n   Numeric cols    : {len(num_cols)}")
print(f"   Categorical cols: {len(cat_cols)}")
print(f"   ID cols         : {id_cols}")
print(f"   Constant cols   : {const_cols}")

# ── Column profiles ────────────────────────────────────────────
print("\n📊 Profiling columns...")
columns = []
for col in df.columns:
    try:
        raw_samples = df[col].dropna().head(5).tolist()
        samples = [
            int(v) if isinstance(v, (np.integer,))
            else float(v) if isinstance(v, (np.floating,))
            else str(v)
            for v in raw_samples
        ]
    except Exception:
        samples = []

    columns.append({
        "name": col,
        "dtype": str(df[col].dtype),
        "null_count": int(df[col].isnull().sum()),
        "null_pct": round(float(df[col].isnull().mean() * 100), 2),
        "unique_count": int(df[col].nunique()),
        "sample_values": samples,
        "is_target": col == TARGET_COL,
        "is_id": col in id_cols,
        "is_constant": col in const_cols
    })

# ── Numeric statistics ─────────────────────────────────────────
print("📊 Computing numeric statistics...")
numeric_stats = {}
for col in num_cols:
    try:
        numeric_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std":  round(float(df[col].std()), 4),
            "min":  round(float(df[col].min()), 4),
            "max":  round(float(df[col].max()), 4),
            "p25":  round(float(df[col].quantile(0.25)), 4),
            "p50":  round(float(df[col].quantile(0.50)), 4),
            "p75":  round(float(df[col].quantile(0.75)), 4),
        }
    except Exception:
        continue

# ── Data quality issues ────────────────────────────────────────
quality_issues = []

high_missing = [
    col for col in df.columns
    if df[col].isnull().mean() > 0.3
]
if high_missing:
    quality_issues.append(
        f"High missing values (>30%): {high_missing}"
    )

if const_cols:
    quality_issues.append(
        f"Constant columns (useless for modeling): {const_cols}"
    )

if dup_count > 0:
    quality_issues.append(
        f"Duplicate rows detected: {dup_count}"
    )

if id_cols:
    quality_issues.append(
        f"ID columns (should be excluded from features): {id_cols}"
    )

if not quality_issues:
    quality_issues.append("No major quality issues detected ✅")

# ── Narrative ──────────────────────────────────────────────────
minority_pct = round(imbalance_ratio * 100, 1)
narrative = (
    f"The dataset contains {row_count:,} records with {col_count} features. "
    f"The target column '{TARGET_COL}' is a {task_type} problem with "
    f"{'significant class imbalance' if imbalance else 'balanced classes'} "
    f"({minority_pct}% minority class). "
    f"Data quality is {'good — no missing values' if missing_pct == 0 else f'moderate — {missing_pct}% missing'}. "
    f"Key preprocessing needs: drop {len(id_cols)} ID columns and "
    f"{len(const_cols)} constant columns before modeling."
)

print(f"\n  📝 Narrative: {narrative[:80]}...")

# ── Save output ────────────────────────────────────────────────
output = {
    "row_count": row_count,
    "column_count": col_count,
    "inferred_task_type": task_type,
    "overall_missing_pct": missing_pct,
    "duplicate_row_count": dup_count,
    "target_distribution": target_dist,
    "class_imbalance_detected": imbalance,
    "imbalance_ratio": imbalance_ratio,
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "id_columns": id_cols,
    "text_columns": text_cols,
    "constant_columns": const_cols,
    "columns": columns,
    "numeric_stats": numeric_stats,
    "quality_issues": quality_issues,
    "confidence_score": 0.95,
    "genai_narrative": narrative
}

with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to: {OUTPUT_PATH}")
print(f"{'='*55}\n")