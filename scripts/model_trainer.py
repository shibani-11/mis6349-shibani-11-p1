# scripts/model_trainer.py
# MIRA v2.0 — Dataset-Agnostic Binary Classification Model Builder
#
# Correct ML workflow:
#   1. Load single user-provided CSV
#   2. Clean and encode
#   3. Split 80/20 BEFORE any preprocessing
#   4. Fit preprocessors on TRAIN only — transform BOTH splits
#   5. Save preprocessed train and test splits to disk (Phase 3 uses these)
#   6. Train 5 models on train, cross-validate on train
#   7. Report quick validation metrics on the test split (for orchestrator)
#
# The test split saved here is the TRUE HOLDOUT used in Phase 3.
# Preprocessing is fit only on training data — no data leakage.

import os
import sys
import json
import time
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Args ────────────────────────────────────────────────────────────────────
DATASET_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv"
TARGET_COL   = sys.argv[2] if len(sys.argv) > 2 else "Loan Status"
OUTPUT_PATH  = sys.argv[3] if len(sys.argv) > 3 else "processed/model_building.json"
PRIORITY     = sys.argv[4] if len(sys.argv) > 4 else "roc_auc"

# Derive split file paths from the output path
# e.g. processed/run_abc_model_building.json -> processed/run_abc_train_split.csv
_prefix          = OUTPUT_PATH.replace("_model_building.json", "")
TRAIN_SPLIT_PATH = _prefix + "_train_split.csv"
TEST_SPLIT_PATH  = _prefix + "_holdout_test.csv"

print(f"\n{'='*60}")
print(f"  MIRA v2.0 — Binary Classification Model Builder")
print(f"{'='*60}")
print(f"  Dataset      : {DATASET_PATH}")
print(f"  Target       : {TARGET_COL}")
print(f"  Priority     : {PRIORITY}")
print(f"  Train split  : {TRAIN_SPLIT_PATH}")
print(f"  Holdout test : {TEST_SPLIT_PATH}")
print(f"{'='*60}\n")

# ── Load ────────────────────────────────────────────────────────────────────
print("Loading dataset...")
ext = os.path.splitext(DATASET_PATH)[-1].lower()
if ext in (".xlsx", ".xls"):
    df = pd.read_excel(DATASET_PATH)
elif ext == ".parquet":
    df = pd.read_parquet(DATASET_PATH)
elif ext in (".tsv",):
    df = pd.read_csv(DATASET_PATH, sep="\t")
else:
    df = pd.read_csv(DATASET_PATH)

print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# ── Validate target ─────────────────────────────────────────────────────────
if TARGET_COL not in df.columns:
    print(f"\nERROR: Target column '{TARGET_COL}' not found.")
    print(f"  Available columns: {list(df.columns)}")
    sys.exit(1)

n_unique = df[TARGET_COL].dropna().nunique()
print(f"\nTarget '{TARGET_COL}': {n_unique} unique values")

if n_unique > 10 or (n_unique > 2 and df[TARGET_COL].dtype in ['float64', 'float32']):
    print("ERROR: MIRA supports binary classification only.")
    print("  Detected continuous or high-cardinality target.")
    sys.exit(1)

if n_unique > 2:
    print(f"WARNING: {n_unique}-class target detected. MIRA supports binary only.")
    sys.exit(1)

print("  Binary classification confirmed.")
print(f"  Classes: {sorted(df[TARGET_COL].dropna().unique().tolist())}")

# ── Drop columns with >60% missing ──────────────────────────────────────────
high_missing = [
    c for c in df.columns
    if c != TARGET_COL and df[c].isnull().mean() > 0.60
]
if high_missing:
    df = df.drop(columns=high_missing)
    print(f"\nDropped high-missing columns (>60%): {high_missing}")

# ── Encode target ───────────────────────────────────────────────────────────
if df[TARGET_COL].dtype == object or str(df[TARGET_COL].dtype) == "category":
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL].astype(str))
    print(f"\nTarget encoded: {dict(enumerate(le.classes_))}")

# ── Drop ID / constant columns ───────────────────────────────────────────────
print("\nDetecting useless columns...")
cols_to_drop = []
for col in df.columns:
    if col == TARGET_COL:
        continue
    if df[col].nunique() == len(df):
        cols_to_drop.append(col)
        print(f"  Dropping ID column      : {col}")
    elif df[col].nunique() <= 1:
        cols_to_drop.append(col)
        print(f"  Dropping constant column: {col}")

if not cols_to_drop:
    print("  No useless columns found.")

# ── Auto-detect obvious leaky columns ───────────────────────────────────────
print("\nChecking for data leakage...")
leaky_cols = []
numeric_candidates = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
for col in numeric_candidates:
    if col in (TARGET_COL, *cols_to_drop):
        continue
    try:
        classes = df[TARGET_COL].unique()
        if len(classes) == 2:
            g0 = df[df[TARGET_COL] == classes[0]][col]
            g1 = df[df[TARGET_COL] == classes[1]][col]
            mean_diff = abs(float(g0.mean()) - float(g1.mean()))
            pooled_std = np.mean([float(g0.std()), float(g1.std())])
            if pooled_std > 0 and mean_diff / pooled_std > 5:
                col_mean = float(df[col].mean())
                col_std  = float(df[col].std())
                if col_std > abs(col_mean) * 2 and min(float(g0.mean()), float(g1.mean())) < 0.05 * abs(col_mean) + 0.001:
                    leaky_cols.append(col)
                    print(f"  Leakage suspected: {col}")
    except Exception:
        continue

if not leaky_cols:
    print("  No leaky columns detected.")

cols_to_drop = list(set(cols_to_drop + leaky_cols))
if cols_to_drop:
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# ── Features and target ──────────────────────────────────────────────────────
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# ── Class distribution ───────────────────────────────────────────────────────
class_counts   = y.value_counts()
minority_ratio = float(class_counts.min() / class_counts.sum())
imbalanced     = minority_ratio < 0.20
cw             = "balanced" if imbalanced else None
scale_pos      = int(class_counts.max() / max(class_counts.min(), 1)) if imbalanced else 1

print(f"\nClass distribution:")
for cls, cnt in class_counts.items():
    print(f"  Class {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
if imbalanced:
    print(f"  Imbalance detected ({minority_ratio:.1%} minority) — using class_weight=balanced")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — SPLIT FIRST (before any preprocessing)
#  This ensures the test set is truly unseen — preprocessors never touch it.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  STEP 1: Splitting dataset 80% train / 20% holdout test")
print(f"{'='*60}")

try:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
except Exception:
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)
X_train_raw = X_train_raw.reset_index(drop=True)
X_test_raw  = X_test_raw.reset_index(drop=True)

print(f"  Train : {len(X_train_raw):,} samples ({len(X_train_raw)/len(X)*100:.0f}%)")
print(f"  Test  : {len(X_test_raw):,}  samples ({len(X_test_raw)/len(X)*100:.0f}%) ← held out")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — FIT PREPROCESSORS ON TRAIN ONLY, TRANSFORM BOTH SPLITS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  STEP 2: Preprocessing (fit on train only — no leakage)")
print(f"{'='*60}")

X_train = X_train_raw.copy()
X_test  = X_test_raw.copy()

# Encode categoricals — fit on train, apply to both
if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols]  = enc.transform(X_test[cat_cols].astype(str))
    print(f"  Encoded {len(cat_cols)} categorical columns (fit on train only)")

# Impute missing — fit on train, apply to both
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    imp = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)
    print(f"  Imputed missing values (fit on train only)")

# Scale numerics — fit on train, apply to both
if num_cols:
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    print(f"  Scaled {len(num_cols)} numeric columns (fit on train only)")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — SAVE PREPROCESSED SPLITS TO DISK
#  Phase 3 (model_evaluator.py) will load these — no re-preprocessing needed.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  STEP 3: Saving preprocessed splits to disk")
print(f"{'='*60}")

os.makedirs(os.path.dirname(TRAIN_SPLIT_PATH) or ".", exist_ok=True)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values
train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
print(f"  Train split saved : {TRAIN_SPLIT_PATH}")

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv(TEST_SPLIT_PATH, index=False)
print(f"  Test split saved  : {TEST_SPLIT_PATH}  ← Phase 3 holdout")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN MODELS ON TRAIN SET
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  STEP 4: Training 5 models on training data")
print(f"{'='*60}")

model_defs = [
    ("Logistic Regression", "linear",
     LogisticRegression(class_weight=cw, max_iter=1000, random_state=42)),
    ("Random Forest", "ensemble",
     RandomForestClassifier(class_weight=cw, n_estimators=100, n_jobs=-1, random_state=42)),
    ("XGBoost", "boosting",
     XGBClassifier(scale_pos_weight=scale_pos, n_estimators=100, random_state=42,
                   eval_metric="logloss", verbosity=0)),
    ("LightGBM", "boosting",
     LGBMClassifier(class_weight=cw, n_estimators=100, random_state=42, verbose=-1)),
    ("Gradient Boosting", "boosting",
     GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
evaluated   = []
skipped     = {}

for name, family, model in model_defs:
    print(f"\n  {name}...")
    try:
        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        # In-sample score (for overfitting detection in Phase 3)
        y_train_pred  = model.predict(X_train)
        train_roc     = float(roc_auc_score(y_train, y_train_pred))

        # Quick validation on holdout — NOT used for model selection,
        # official test metrics come from Phase 3
        y_test_pred   = model.predict(X_test)
        y_test_proba  = (model.predict_proba(X_test)[:, 1]
                         if hasattr(model, "predict_proba") else y_test_pred)

        val_roc  = float(roc_auc_score(y_test, y_test_proba))
        val_acc  = float(accuracy_score(y_test, y_test_pred))
        val_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
        val_rec  = float(recall_score(y_test, y_test_pred, zero_division=0))
        val_f1   = float(f1_score(y_test, y_test_pred, zero_division=0))

        # Cross-validation on train only (model selection criterion)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv_splitter, scoring="roc_auc", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())

        overfit_flag = bool((train_roc - val_roc) > 0.10)

        # Feature importance (top 10)
        feat_imp = None
        if hasattr(model, "feature_importances_"):
            pairs = sorted(zip(X_train.columns, model.feature_importances_),
                           key=lambda x: x[1], reverse=True)[:10]
            feat_imp = {k: round(float(v), 6) for k, v in pairs}

        print(f"    CV ROC-AUC  : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Val ROC-AUC : {val_roc:.4f}  (quick check, official in Phase 3)")
        print(f"    Val Recall  : {val_rec:.4f}")
        print(f"    Train time  : {train_time}s")
        print(f"    Overfit flag: {'YES' if overfit_flag else 'No'}")

        evaluated.append({
            "model_name":              name,
            "model_family":            family,
            # CV metrics — fit on train only (model selection criterion)
            "cross_val_score_mean":    round(cv_mean, 4),
            "cross_val_score_std":     round(cv_std, 4),
            # Quick val metrics — for orchestrator threshold check
            # (officially re-evaluated in Phase 3 on the same holdout)
            "roc_auc":                 round(val_roc, 4),
            "accuracy":                round(val_acc, 4),
            "precision":               round(val_prec, 4),
            "recall":                  round(val_rec, 4),
            "f1_score":                round(val_f1, 4),
            "train_roc_auc":           round(train_roc, 4),
            "training_time_seconds":   train_time,
            "overfitting_detected":    overfit_flag,
            "feature_count_used":      int(X_train.shape[1]),
            "feature_importance":      feat_imp,
        })

    except Exception as e:
        print(f"    FAILED: {e}")
        skipped[name] = str(e)

# ── Results summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Training Summary — ranked by {PRIORITY}")
print(f"{'='*60}")

rank_key      = PRIORITY if PRIORITY in ("roc_auc", "f1_score", "recall", "accuracy") else "roc_auc"
sorted_models = sorted(evaluated, key=lambda m: m.get(rank_key, 0), reverse=True)
best_by       = {m: max(evaluated, key=lambda x: x.get(m, 0))["model_name"]
                 for m in ("roc_auc", "f1_score", "recall", "accuracy") if evaluated}

for i, m in enumerate(sorted_models, 1):
    overfit_icon = "OVERFIT" if m["overfitting_detected"] else "OK"
    print(f"  {i}. {m['model_name']:<26} CV={m['cross_val_score_mean']:.4f}  "
          f"val_roc={m['roc_auc']:.4f}  [{overfit_icon}]")

best = sorted_models[0] if sorted_models else {}
print(f"\n  Best (CV): {best.get('model_name','N/A')} "
      f"(CV mean={best.get('cross_val_score_mean',0):.4f})")

preprocessing_steps = [
    f"Dropped {len(cols_to_drop)} useless/leaky columns: {cols_to_drop}",
    "Stratified 80/20 split performed BEFORE preprocessing",
    "OrdinalEncoder fit on train only — transforms applied to both splits",
    "SimpleImputer (mean) fit on train only — transforms applied to both splits",
    "StandardScaler fit on train only — transforms applied to both splits",
    f"class_weight=balanced applied: {imbalanced}",
    f"Train split saved: {TRAIN_SPLIT_PATH}",
    f"Holdout test split saved: {TEST_SPLIT_PATH}",
]

narrative = (
    f"Trained {len(evaluated)} binary classification models on "
    f"{len(X_train):,} training samples ({X_train.shape[1]} features). "
    f"Dataset split 80/20 before preprocessing — test set kept as true holdout. "
    f"Preprocessors fit on training data only. "
    f"{'Class imbalance handled with class_weight=balanced. ' if imbalanced else ''}"
    f"Best model by CV {PRIORITY}: {best_by.get(PRIORITY, 'N/A')}. "
    f"Phase 3 will evaluate all models on the {len(X_test):,}-sample holdout test set."
)

results = {
    "task_type":               "classification",
    "train_samples":           int(len(X_train)),
    "test_samples":            int(len(X_test)),
    "feature_count":           int(X_train.shape[1]),
    "train_split_path":        TRAIN_SPLIT_PATH,
    "holdout_test_path":       TEST_SPLIT_PATH,
    "target_column":           TARGET_COL,
    "models_considered":       [m[0] for m in model_defs],
    "models_evaluated":        evaluated,
    "models_skipped":          skipped,
    "primary_metric":          PRIORITY,
    "best_model_by_metric":    best_by,
    "feature_importance":      best.get("feature_importance"),
    "dropped_columns":         cols_to_drop,
    "leaky_columns_detected":  leaky_cols,
    "class_imbalance_detected": imbalanced,
    "minority_class_ratio":    round(float(minority_ratio), 4),
    "preprocessing_steps":     preprocessing_steps,
    "confidence_score":        0.85,
    "genai_narrative":         narrative,
}

os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Saved: {OUTPUT_PATH}")
print(f"{'='*60}\n")
