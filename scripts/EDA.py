"""
EDA.py — Phase 1: Data Cleaning, Exploration, and Pre-Modeling.
Writes data_card.json and a cleaned+encoded+scaled CSV for the training script.
"""
import argparse, json, pathlib, warnings, os
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv
import litellm

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",          required=True, help="Path to raw CSV dataset")
parser.add_argument("--target",           required=True, help="Target column name")
parser.add_argument("--output",           required=True, help="Path to write data_card.json")
parser.add_argument("--cleaned-output",   required=True, help="Path to write cleaned+processed CSV")
parser.add_argument("--business-problem", required=False, default="", help="Business problem description")
args = parser.parse_args()

target = args.target
df = pd.read_csv(args.dataset)

cleaning_log    = []   # human-readable steps taken
data_quality_issues = []

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════
print("\n── DATA CLEANING ──")

# 1. Shape and dtypes
original_shape = df.shape
print(f"  Shape: {original_shape[0]:,} rows × {original_shape[1]} columns")
dtype_map = df.dtypes.astype(str).to_dict()

# 2. Handle missing values
null_counts = df.isnull().sum()
missing_cols = null_counts[null_counts > 0]
missing_value_summary = {}
for col in missing_cols.index:
    pct = missing_cols[col] / len(df)
    if pct > 0.50:
        df.drop(columns=[col], inplace=True)
        missing_value_summary[col] = int(missing_cols[col])
        cleaning_log.append(f"Dropped '{col}' ({pct:.0%} missing)")
        data_quality_issues.append(f"'{col}' dropped — {pct:.0%} missing values")
    elif df[col].dtype == "object":
        fill_val = df[col].mode()[0]
        df[col].fillna(fill_val, inplace=True)
        missing_value_summary[col] = int(missing_cols[col])
        cleaning_log.append(f"Imputed '{col}' with mode='{fill_val}' ({missing_cols[col]} nulls)")
    else:
        fill_val = df[col].median()
        df[col].fillna(fill_val, inplace=True)
        missing_value_summary[col] = int(missing_cols[col])
        cleaning_log.append(f"Imputed '{col}' with median={fill_val:.2f} ({missing_cols[col]} nulls)")

if missing_value_summary:
    print(f"  Missing values handled: {list(missing_value_summary.keys())}")
    data_quality_issues.append(f"Missing values found and imputed in: {list(missing_value_summary.keys())}")
else:
    print("  No missing values")

# 3. Remove duplicates
n_dupes = int(df.duplicated().sum())
if n_dupes > 0:
    df.drop_duplicates(inplace=True)
    cleaning_log.append(f"Removed {n_dupes} duplicate rows")
    data_quality_issues.append(f"{n_dupes} duplicate rows removed")
    print(f"  Removed {n_dupes} duplicate rows")
else:
    print("  No duplicates")

# 4. Drop ID and constant columns
ID_KEYWORDS = {"id", "rownumber", "customerid", "rowid", "uuid", "surname", "name", "index"}
id_cols      = [c for c in df.columns if c.lower() in ID_KEYWORDS and c != target]
const_cols   = [c for c in df.columns if df[c].nunique() <= 1 and c != target]
drop_these   = sorted(set(id_cols + const_cols))
if drop_these:
    df.drop(columns=[c for c in drop_these if c in df.columns], inplace=True)
    cleaning_log.append(f"Dropped ID/constant columns: {drop_these}")
    print(f"  Dropped ID/constant columns: {drop_these}")

# 5. Fix data types (object columns that are actually numeric)
for col in df.select_dtypes(include=["object"]).columns:
    if col == target:
        continue
    try:
        df[col] = pd.to_numeric(df[col])
        cleaning_log.append(f"Converted '{col}' from object → numeric")
        print(f"  Converted '{col}' to numeric")
    except (ValueError, TypeError):
        pass

# 6. Handle outliers — IQR capping (Winsorization)
outlier_report = {}
for col in df.select_dtypes(include="number").columns:
    if col == target:
        continue
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        continue
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
    if n_out > 0:
        df[col] = df[col].clip(lower=lo, upper=hi)
        outlier_report[col] = n_out
if outlier_report:
    cleaning_log.append(f"IQR outlier capping applied to: {list(outlier_report.keys())}")
    data_quality_issues.append(f"Outliers capped (IQR method) in: {list(outlier_report.keys())}")
    print(f"  Outliers capped in: {list(outlier_report.keys())}")
else:
    print("  No outliers above IQR threshold")

# 7. Standardize categories — strip whitespace + title case
str_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in str_cols:
    if col == target:
        continue
    df[col] = df[col].str.strip().str.title()
if str_cols:
    cleaning_log.append(f"Standardized string columns (strip + title): {str_cols}")


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════
print("\n── DATA EXPLORATION ──")

# Target distribution + class imbalance
class_dist           = df[target].value_counts(normalize=True).sort_index()
class_distribution   = {str(k): round(float(v), 4) for k, v in class_dist.items()}
minority_class_ratio = float(min(class_distribution.values()))
class_imbalance_detected = minority_class_ratio < 0.20
minority_pct = round(minority_class_ratio * 100, 1)
print(f"  Target distribution: {class_distribution}")
print(f"  Imbalance: {class_imbalance_detected} (minority={minority_pct}%)")

# Numeric distributions
num_cols_exp = [c for c in df.select_dtypes(include="number").columns if c != target]
numeric_summary = df[num_cols_exp].describe().round(4).to_dict() if num_cols_exp else {}

# Categorical distributions
cat_cols_raw = [c for c in df.select_dtypes(include="object").columns if c != target]
cat_summary = {col: df[col].value_counts().head(5).to_dict() for col in cat_cols_raw}
if cat_summary:
    print(f"  Categorical columns: {list(cat_summary.keys())}")

# Correlations with target (point-biserial / Pearson)
numeric_df = df.select_dtypes(include="number")
if target in numeric_df.columns and len(numeric_df.columns) > 1:
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    high_correlation_features = [
        {"feature": feat, "correlation": round(float(val), 4)}
        for feat, val in corr.head(5).items()
    ]
else:
    high_correlation_features = []
print(f"  Top correlated features: {[f['feature'] for f in high_correlation_features[:3]]}")

# 8. Leaky column detection (single-feature AUC > 0.98 vs target)
leaky_cols = []
if df[target].nunique() == 2:
    y_check = df[target]
    for col in df.select_dtypes(include="number").columns:
        if col == target:
            continue
        try:
            score = roc_auc_score(y_check, df[col])
            score = max(score, 1 - score)
            if score > 0.98:
                leaky_cols.append(col)
        except Exception:
            pass
    if leaky_cols:
        df.drop(columns=leaky_cols, inplace=True)
        cleaning_log.append(f"Removed leaky columns (single-feature AUC > 0.98): {leaky_cols}")
        data_quality_issues.append(f"Leaky columns removed: {leaky_cols}")
        print(f"  Leaky columns removed: {leaky_cols}")
    else:
        print("  No data leakage detected in individual features")


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — PRE-MODELING
# ══════════════════════════════════════════════════════════════════
print("\n── PRE-MODELING ──")

# Encode categoricals
cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target]
le_encodings = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_encodings[col] = {str(cls): int(idx) for cls, idx in zip(le.classes_, le.transform(le.classes_))}
if cat_cols:
    print(f"  Label-encoded: {cat_cols}")

# Scale numerics
X_raw = df.drop(columns=[target])
y     = df[target]
numeric_to_scale = X_raw.select_dtypes(include="number").columns.tolist()
scaler = StandardScaler()
X_scaled = X_raw.copy()
X_scaled[numeric_to_scale] = scaler.fit_transform(X_raw[numeric_to_scale])
print(f"  StandardScaler applied to {len(numeric_to_scale)} numeric features")

# Stratified train/val split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} rows  Val: {len(X_val):,} rows  (stratified)")

# Save cleaned + encoded + scaled data (full set — training script will do its own CV splits)
cleaned_df = X_scaled.copy()
cleaned_df[target] = y.values
pathlib.Path(args.cleaned_output).parent.mkdir(parents=True, exist_ok=True)
cleaned_df.to_csv(args.cleaned_output, index=False)
print(f"  Cleaned data saved → {args.cleaned_output}")

# Build recommended_approach string
approach_parts = []
if drop_these:
    approach_parts.append(f"dropped ID columns {drop_these}")
if cat_cols:
    approach_parts.append(f"label-encoded categoricals {cat_cols}")
if numeric_to_scale:
    approach_parts.append(f"StandardScaler on {len(numeric_to_scale)} numeric features")
if class_imbalance_detected:
    approach_parts.append("class_weight='balanced' recommended (minority < 20%)")
recommended_approach = "; ".join(approach_parts) if approach_parts else "standard preprocessing applied"

# GenAI narrative
top_feat = high_correlation_features[0]["feature"] if high_correlation_features else "unknown"
imbalance_note = (
    f"Class imbalance detected ({minority_pct}% minority) — balanced weighting required."
    if class_imbalance_detected else
    f"Classes are reasonably balanced ({minority_pct}% minority)."
)
genai_narrative = (
    f"The dataset contains {len(df):,} records across {len(df.columns) - 1} features after cleaning. "
    f"{imbalance_note} "
    f"The strongest predictor of the target is '{top_feat}'. "
    f"Data is cleaned, encoded, and scaled — ready for 5-fold cross-validated model training."
)

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — INFER PRIORITY METRIC FROM BUSINESS PROBLEM
# ══════════════════════════════════════════════════════════════════
print("\n── METRIC INFERENCE ──")

VALID_METRICS = ["roc_auc", "recall", "f1_score", "precision"]

def infer_priority_metric(business_problem: str, imbalance: bool, minority_ratio: float) -> tuple[str, str]:
    """Ask the LLM to pick the best evaluation metric given the business problem."""
    if not business_problem.strip():
        return "roc_auc", "No business problem provided — defaulting to roc_auc."

    prompt = f"""You are an ML metric selection expert.

Business problem:
\"\"\"{business_problem}\"\"\"

Dataset facts:
- Class imbalance detected: {imbalance}
- Minority class ratio: {minority_ratio:.3f}

Choose the single best evaluation metric for model selection from this list:
- roc_auc    : best for balanced datasets and general ranking ability
- recall     : best when missing a positive case (false negative) is costly
- f1_score   : best when both false positives and false negatives matter equally
- precision  : best when false positives are very costly

Respond with a JSON object with exactly two keys:
{{
  "metric": "<one of: roc_auc, recall, f1_score, precision>",
  "reason": "<one sentence explaining why this metric fits the business problem>"
}}

No markdown. No explanation outside the JSON."""

    try:
        response = litellm.completion(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=os.getenv("LLM_API_KEY"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        metric = parsed.get("metric", "roc_auc")
        reason = parsed.get("reason", "")
        if metric not in VALID_METRICS:
            metric = "roc_auc"
            reason = f"LLM returned unknown metric — defaulted to roc_auc."
        return metric, reason
    except Exception as e:
        return "roc_auc", f"Metric inference failed ({e}) — defaulted to roc_auc."

priority_metric, metric_reason = infer_priority_metric(
    args.business_problem, class_imbalance_detected, minority_class_ratio
)
print(f"  Priority metric: {priority_metric}")
print(f"  Reason: {metric_reason}")

# ══════════════════════════════════════════════════════════════════
# OUTPUT — data_card.json
# ══════════════════════════════════════════════════════════════════
out = {
    "rows":                      len(df),
    "features":                  len(df.columns) - 1,
    "class_distribution":        class_distribution,
    "class_imbalance_detected":  class_imbalance_detected,
    "minority_class_ratio":      minority_class_ratio,
    "missing_value_summary":     missing_value_summary,
    "high_correlation_features": high_correlation_features,
    "data_quality_issues":       data_quality_issues,
    "recommended_approach":      recommended_approach,
    "genai_narrative":           genai_narrative,
    # Metadata for downstream scripts
    "cleaned_data_path":         args.cleaned_output,
    "encoded_columns":           cat_cols,
    "scaled_columns":            numeric_to_scale,
    "leaky_columns_removed":     leaky_cols,
    "dropped_columns":           drop_these,
    "cleaning_log":              cleaning_log,
    "priority_metric":           priority_metric,
    "metric_reason":             metric_reason,
}

REQUIRED_KEYS = [
    "rows", "features", "class_distribution", "class_imbalance_detected",
    "minority_class_ratio", "missing_value_summary", "high_correlation_features",
    "data_quality_issues", "recommended_approach", "genai_narrative",
]
missing = [k for k in REQUIRED_KEYS if k not in out]
if missing:
    raise ValueError(f"SCHEMA VIOLATION — missing keys: {missing}")

pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)

print("\nSCHEMA OK")
print(f"  rows={len(df)}  features={len(df.columns)-1}  minority_ratio={minority_class_ratio:.3f}  imbalance={class_imbalance_detected}")
print(f"  top_feature={top_feat}  quality_issues={len(data_quality_issues)}  priority_metric={priority_metric}")
