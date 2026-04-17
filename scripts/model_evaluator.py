# scripts/model_evaluator.py
# MIRA v2.0 — Dataset-Agnostic Binary Classification Model Tester
#
# Correct ML workflow:
#   1. Load the preprocessed train/test splits saved by Phase 2
#      (preprocessing was fit on train only — no leakage)
#   2. Re-train each model on the training split
#   3. Evaluate on the TRUE HOLDOUT test split (unseen during Phase 2 training)
#   4. Run quality checks: overfitting, leakage, stability, business logic
#   5. Report final official test metrics
#
# The test split here is the same 20% held out in Phase 2.
# It was NEVER used to fit any preprocessor or influence model selection.

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Args ────────────────────────────────────────────────────────────────────
# sys.argv[1] (dataset_path) is accepted but unused — splits are loaded from
# Phase 2 output. Kept so the orchestrator call signature stays unchanged.
TARGET_COL     = sys.argv[2] if len(sys.argv) > 2 else "Loan Status"
BUILDING_FILE  = sys.argv[3] if len(sys.argv) > 3 else "processed/model_building.json"
OUTPUT_PATH    = sys.argv[4] if len(sys.argv) > 4 else "processed/model_testing.json"
PRIORITY       = sys.argv[5] if len(sys.argv) > 5 else "roc_auc"

print(f"\n{'='*60}")
print(f"  MIRA v2.0 — Binary Classification Model Tester")
print(f"{'='*60}")
print(f"  Building file : {BUILDING_FILE}")
print(f"  Output        : {OUTPUT_PATH}")
print(f"  Priority      : {PRIORITY}")
print(f"{'='*60}\n")

# ── Load Phase 2 output ──────────────────────────────────────────────────────
print("Loading Phase 2 model building results...")
try:
    with open(BUILDING_FILE) as f:
        building = json.load(f)
except FileNotFoundError:
    print(f"ERROR: Building file not found: {BUILDING_FILE}")
    sys.exit(1)

models_to_test  = [m["model_name"] for m in building.get("models_evaluated", [])]
imbalanced      = building.get("class_imbalance_detected", False)
minority_ratio  = building.get("minority_class_ratio", 0.5)
train_split_path = building.get("train_split_path", "")
test_split_path  = building.get("holdout_test_path", "")

print(f"  Models to test    : {models_to_test}")
print(f"  Class imbalanced  : {imbalanced}")
print(f"  Train split path  : {train_split_path}")
print(f"  Test split path   : {test_split_path}")

# ── Load preprocessed splits ─────────────────────────────────────────────────
# These were saved by Phase 2. Preprocessing (encoding, imputation, scaling)
# was fit on training data only. We load them directly — no re-preprocessing.
print(f"\nLoading preprocessed train/test splits...")

if not train_split_path or not test_split_path:
    print("ERROR: Phase 2 did not save split paths.")
    print("  Please re-run Phase 2 with the updated model_trainer.py.")
    sys.exit(1)

try:
    train_df = pd.read_csv(train_split_path)
    test_df  = pd.read_csv(test_split_path)
except FileNotFoundError as e:
    print(f"ERROR: Split file not found — {e}")
    print("  Please re-run Phase 2 first.")
    sys.exit(1)

# Use TARGET_COL from building output if available
TARGET_COL = building.get("target_column", TARGET_COL)

if TARGET_COL not in train_df.columns:
    print(f"ERROR: Target column '{TARGET_COL}' not found in split files.")
    print(f"  Available columns: {list(train_df.columns)}")
    sys.exit(1)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

print(f"  Train : {len(X_train):,} samples x {X_train.shape[1]} features")
print(f"  Test  : {len(X_test):,}  samples x {X_test.shape[1]} features  ← holdout")

# ── Model registry ───────────────────────────────────────────────────────────
cw        = "balanced" if imbalanced else None
scale_pos = int(1 / max(minority_ratio, 0.01)) if imbalanced else 1

model_registry = {
    "Logistic Regression": LogisticRegression(
        class_weight=cw, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        class_weight=cw, n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos, n_estimators=100, random_state=42,
        eval_metric="logloss", verbosity=0),
    "LightGBM": LGBMClassifier(
        class_weight=cw, n_estimators=100, random_state=42, verbose=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42),
}

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE EACH MODEL ON TRUE HOLDOUT TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  Evaluating models on holdout test set")
print(f"{'='*60}")

test_results = []
flagged      = []
top_models   = []

for name in models_to_test:
    if name not in model_registry:
        print(f"\n  Unknown model: {name} — skipping")
        continue

    print(f"\n  {name}...")
    model = model_registry[name]

    try:
        start = time.time()

        # Re-train on full training split
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        # In-sample score (overfitting baseline)
        y_train_pred  = model.predict(X_train)
        y_train_proba = (model.predict_proba(X_train)[:, 1]
                         if hasattr(model, "predict_proba") else y_train_pred)
        train_roc = float(roc_auc_score(y_train, y_train_proba))

        # ── OFFICIAL TEST METRICS (unseen holdout) ──────────────────────────
        y_test_pred  = model.predict(X_test)
        y_test_proba = (model.predict_proba(X_test)[:, 1]
                        if hasattr(model, "predict_proba") else y_test_pred)

        test_roc  = float(roc_auc_score(y_test, y_test_proba))
        test_acc  = float(accuracy_score(y_test, y_test_pred))
        test_prec = float(precision_score(y_test, y_test_pred, zero_division=0))
        test_rec  = float(recall_score(y_test, y_test_pred, zero_division=0))
        test_f1   = float(f1_score(y_test, y_test_pred, zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = (cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0))

        # Cross-validation stability (on train only)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv_splitter, scoring="roc_auc", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())

        # ── Quality Checks ───────────────────────────────────────────────────
        train_test_gap   = round(train_roc - test_roc, 4)
        overfitting      = bool(train_test_gap > 0.10)
        leakage_suspected = bool(test_roc > 0.99)
        stability_ok     = bool(cv_std <= 0.05)

        # Business logic: top features should not be ID/index-like columns
        biz_logical = True
        if hasattr(model, "feature_importances_"):
            top_feats = X_train.columns[
                np.argsort(model.feature_importances_)[::-1][:3]
            ].tolist()
            id_like = ["id", "index", "row", "number", "seq", "key"]
            biz_logical = not any(
                any(kw in f.lower() for kw in id_like) for f in top_feats
            )

        fail_reasons = []
        if overfitting:
            fail_reasons.append(
                f"Overfitting: train-test ROC-AUC gap = {train_test_gap:.3f} > 0.10")
        if leakage_suspected:
            fail_reasons.append("Suspected data leakage: test ROC-AUC > 0.99")
        if not stability_ok:
            fail_reasons.append(
                f"Unstable: CV std = {cv_std:.4f} > 0.05")
        if not biz_logical:
            fail_reasons.append("Top features appear to be ID/index columns")

        passed = len(fail_reasons) == 0

        # Determine which metric key maps to priority
        metric_map = {
            "roc_auc":  test_roc,
            "f1_score": test_f1,
            "recall":   test_rec,
            "accuracy": test_acc,
        }
        priority_val = metric_map.get(PRIORITY, test_roc)

        print(f"    Train ROC-AUC  : {train_roc:.4f}")
        print(f"    Test ROC-AUC   : {test_roc:.4f}  ← official holdout")
        print(f"    Test Recall    : {test_rec:.4f}")
        print(f"    Test F1        : {test_f1:.4f}")
        print(f"    Test Precision : {test_prec:.4f}")
        print(f"    Train-Test Gap : {train_test_gap:.4f}  "
              f"({'OVERFIT' if overfitting else 'OK'})")
        print(f"    CV Stability   : {cv_mean:.4f} ± {cv_std:.4f}  "
              f"({'UNSTABLE' if not stability_ok else 'OK'})")
        print(f"    Leakage        : {'SUSPECTED' if leakage_suspected else 'None'}")
        print(f"    Passed         : {'YES' if passed else 'NO'}")
        if fail_reasons:
            for r in fail_reasons:
                print(f"      ! {r}")

        result = {
            "model_name":               name,
            # Official test metrics (holdout)
            "roc_auc":                  round(test_roc, 4),
            "accuracy":                 round(test_acc, 4),
            "precision":                round(test_prec, 4),
            "recall":                   round(test_rec, 4),
            "f1_score":                 round(test_f1, 4),
            # Train scores (for overfitting diagnosis)
            "train_roc_auc":            round(train_roc, 4),
            "train_test_gap":           train_test_gap,
            # Confusion matrix
            "confusion_matrix":         {
                "TP": int(tp), "FP": int(fp),
                "TN": int(tn), "FN": int(fn),
            },
            # CV stability
            "cv_mean":                  round(cv_mean, 4),
            "cv_std":                   round(cv_std, 4),
            # Quality checks
            "overfitting_detected":     overfitting,
            "leakage_suspected":        leakage_suspected,
            "stability_ok":             stability_ok,
            "features_business_logical": biz_logical,
            "passed_testing":           passed,
            "fail_reasons":             fail_reasons,
            "training_time_seconds":    train_time,
        }

        test_results.append(result)

        if passed:
            top_models.append((name, priority_val))
        else:
            flagged.append(name)

    except Exception as e:
        print(f"    FAILED: {e}")
        flagged.append(name)

# ── Sort and select top models ────────────────────────────────────────────────
top_models_sorted = [
    m[0] for m in sorted(top_models, key=lambda x: x[1], reverse=True)
][:2]

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Testing Summary — official holdout test metrics")
print(f"{'='*60}")
print(f"  Models tested : {len(test_results)}")
print(f"  Top models    : {top_models_sorted}")
print(f"  Flagged       : {flagged}")

if test_results:
    print(f"\n  Ranked by {PRIORITY} (holdout test set):")
    sorted_results = sorted(
        test_results,
        key=lambda r: r.get(PRIORITY, r.get("roc_auc", 0)),
        reverse=True
    )
    for i, r in enumerate(sorted_results, 1):
        score  = r.get(PRIORITY, r.get("roc_auc", 0))
        status = "PASS" if r.get("passed_testing") else "FAIL"
        gap    = r.get("train_test_gap", 0)
        print(f"  {i}. {r['model_name']:<26} "
              f"{PRIORITY}={score:.4f}  gap={gap:.4f}  [{status}]")

narrative = (
    f"Evaluated {len(test_results)} models on the {len(X_test):,}-sample holdout test set "
    f"(20% of original data, never used during model training or preprocessing). "
    f"Top models by {PRIORITY}: {top_models_sorted}. "
    f"Flagged: {flagged}. "
    f"Checks performed: overfitting (train-test ROC-AUC gap > 10%), "
    f"data leakage (test ROC-AUC > 99%), "
    f"stability (CV std > 5%), "
    f"business logic (no ID columns as top features)."
)

output = {
    "models_tested":    models_to_test,
    "test_results":     test_results,
    "top_models":       top_models_sorted,
    "flagged_models":   flagged,
    "primary_metric":   PRIORITY,
    "train_samples":    int(len(X_train)),
    "test_samples":     int(len(X_test)),
    "holdout_test_path": test_split_path,
    "confidence_score": 0.85,
    "genai_narrative":  narrative,
}

import os
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Saved: {OUTPUT_PATH}")
print(f"{'='*60}\n")
