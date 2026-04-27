"""
Modeltrain.py — Phase 2 + 3: Train all classifiers with 5-fold CV, then stress-test the winner.
Reads cleaned+scaled CSV from EDA.py. Writes model_selection.json.
"""
import argparse, json, pathlib, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--cleaned-data", required=True, help="Cleaned+scaled CSV from EDA.py")
parser.add_argument("--data-card",    required=True, help="data_card.json from EDA.py")
parser.add_argument("--target",       required=True, help="Target column name")
parser.add_argument("--output",       required=True, help="Path to write model_selection.json")
args = parser.parse_args()

with open(args.data_card) as f:
    data_card = json.load(f)

df     = pd.read_csv(args.cleaned_data)
target = args.target

X = df.drop(columns=[target])
y = df[target]

minority_class_ratio = data_card.get("minority_class_ratio", 0.5)
use_balanced = minority_class_ratio < 0.20
cw = "balanced" if use_balanced else None

priority_metric = data_card.get("priority_metric", "roc_auc")
METRIC_KEY = {
    "roc_auc":   "cv_roc_auc_mean",
    "recall":    "cv_recall_mean",
    "f1_score":  "cv_f1_mean",
    "precision": "cv_precision_mean",
}.get(priority_metric, "cv_roc_auc_mean")
print(f"\n  Priority metric: {priority_metric} (ranking key: {METRIC_KEY})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════
candidate_models = [
    ("Logistic Regression",   LogisticRegression(max_iter=3000, solver="saga", class_weight=cw, random_state=42)),
    ("Random Forest",         RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=42, n_jobs=-1)),
    ("Gradient Boosting",     GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)),
    ("XGBoost",               XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, eval_metric="logloss", random_state=42, verbosity=0)),
    ("LightGBM",              LGBMClassifier(n_estimators=200, learning_rate=0.05, class_weight=cw, random_state=42, verbosity=-1)),
]

excluded_models = []
if len(X) > 10_000:
    excluded_models.append({"name": "SVM", "reason": f"Dataset has {len(X):,} rows — SVM excluded above 10k threshold due to O(n²) complexity"})

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — CROSS-VALIDATED TRAINING
# ══════════════════════════════════════════════════════════════════
print("\n── MODEL TRAINING (5-fold stratified CV) ──")

scoring      = ["roc_auc", "f1", "recall", "precision"]
fitted_models = {}
models_trained = []

for name, model in candidate_models:
    print(f"  Training {name}...", end=" ", flush=True)
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)

    # Refit on full set to get train_score and feature importances
    model.fit(X, y)
    fitted_models[name] = model

    train_score     = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    cv_auc_mean     = float(np.mean(cv_res["test_roc_auc"]))
    cv_auc_std      = float(np.std(cv_res["test_roc_auc"]))
    cv_f1_mean      = float(np.mean(cv_res["test_f1"]))
    cv_recall_mean  = float(np.mean(cv_res["test_recall"]))
    cv_prec_mean    = float(np.mean(cv_res["test_precision"]))
    overfitting_gap = round(train_score - cv_auc_mean, 4)

    strengths  = []
    weaknesses = []

    if cv_auc_mean >= 0.85:
        strengths.append(f"Strong AUC ({cv_auc_mean:.4f}) — high discriminative power")
    elif cv_auc_mean >= 0.75:
        strengths.append(f"Good AUC ({cv_auc_mean:.4f})")
    else:
        weaknesses.append(f"Weak AUC ({cv_auc_mean:.4f}) — limited discriminative power")

    if cv_auc_std <= 0.02:
        strengths.append(f"Very stable across folds (std={cv_auc_std:.4f})")
    elif cv_auc_std <= 0.05:
        strengths.append(f"Stable across folds (std={cv_auc_std:.4f})")
    else:
        weaknesses.append(f"High variance across folds (std={cv_auc_std:.4f}) — unstable")

    if abs(overfitting_gap) <= 0.05:
        strengths.append(f"Low overfitting risk (gap={overfitting_gap:.4f})")
    elif abs(overfitting_gap) <= 0.10:
        weaknesses.append(f"Moderate overfitting (gap={overfitting_gap:.4f})")
    else:
        weaknesses.append(f"High overfitting risk (gap={overfitting_gap:.4f} > 0.10 threshold)")

    models_trained.append({
        "name":              name,
        "cv_roc_auc_mean":   round(cv_auc_mean, 4),
        "cv_roc_auc_std":    round(cv_auc_std, 4),
        "cv_f1_mean":        round(cv_f1_mean, 4),
        "cv_recall_mean":    round(cv_recall_mean, 4),
        "cv_precision_mean": round(cv_prec_mean, 4),
        "train_score":       round(train_score, 4),
        "val_score":         round(cv_auc_mean, 4),
        "overfitting_gap":   overfitting_gap,
        "strengths":         strengths,
        "weaknesses":        weaknesses,
    })
    print(f"AUC={cv_auc_mean:.4f} ± {cv_auc_std:.4f}  gap={overfitting_gap:.4f}")

# Sort descending by priority metric inferred from business problem
models_trained.sort(key=lambda m: m[METRIC_KEY], reverse=True)

winner    = models_trained[0]
runner_up = models_trained[1]
rejected  = models_trained[2:]

print(f"\n  Winner:    {winner['name']} ({priority_metric}={winner[METRIC_KEY]:.4f})")
print(f"  Runner-up: {runner_up['name']} ({priority_metric}={runner_up[METRIC_KEY]:.4f})")

rejected_models = [
    {
        "name":               m["name"],
        "cv_roc_auc_mean":    m["cv_roc_auc_mean"],
        "shortfall_vs_winner": round(winner["cv_roc_auc_mean"] - m["cv_roc_auc_mean"], 4),
        "reason": (
            f"AUC {m['cv_roc_auc_mean']:.4f} — "
            f"{winner['cv_roc_auc_mean'] - m['cv_roc_auc_mean']:.4f} below winner; "
            + ("high overfitting risk" if abs(m["overfitting_gap"]) > 0.10 else "insufficient discriminative power")
        ),
    }
    for m in rejected
]

preprocessing_applied = data_card.get("cleaning_log", [])[:5]  # top 5 steps from EDA

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — PHASE 3 STRESS TESTS (winner model)
# ══════════════════════════════════════════════════════════════════
print("\n── STRESS TESTS ──")

winner_model     = fitted_models[winner["name"]]
overfitting_gap  = winner["overfitting_gap"]
overfitting_detected = abs(overfitting_gap) > 0.10
leakage_detected     = winner["cv_roc_auc_mean"] > 0.99
stability_flag       = winner["cv_roc_auc_std"] > 0.05
test_verdict         = "FAIL" if (overfitting_detected or leakage_detected) else "PASS"

test_findings = [
    f"Overfitting check {'FAILED' if overfitting_detected else 'PASSED'}: "
    f"train={winner['train_score']:.4f} − val={winner['val_score']:.4f} = gap={overfitting_gap:.4f} "
    f"({'exceeds' if overfitting_detected else 'within'} 0.10 threshold)",

    f"Leakage check {'FAILED' if leakage_detected else 'PASSED'}: "
    f"best_auc={winner['cv_roc_auc_mean']:.4f} "
    f"({'exceeds' if leakage_detected else 'below'} 0.99 suspicious threshold)",

    f"Stability check {'FLAGGED' if stability_flag else 'PASSED'}: "
    f"cv_std={winner['cv_roc_auc_std']:.4f} "
    f"({'exceeds' if stability_flag else 'within'} 0.05 stability threshold)",
]

for finding in test_findings:
    print(f"  {finding}")
print(f"  Verdict: {test_verdict}")

# Feature importance
model_obj = winner_model
if hasattr(model_obj, "feature_importances_"):
    raw_imp = model_obj.feature_importances_
elif hasattr(model_obj, "coef_"):
    raw_imp = np.abs(model_obj.coef_[0])
else:
    raw_imp = np.zeros(X.shape[1])

feat_pairs = sorted(zip(X.columns.tolist(), raw_imp.tolist()), key=lambda x: x[1], reverse=True)[:10]
feature_importance = {k: round(float(v), 6) for k, v in feat_pairs}

# Selection reasoning
gap_between = winner[METRIC_KEY] - runner_up[METRIC_KEY]
selection_reasoning = (
    f"{winner['name']} achieved the highest cross-validated {priority_metric} of {winner[METRIC_KEY]:.4f} "
    f"(AUC={winner['cv_roc_auc_mean']:.4f}, std={winner['cv_roc_auc_std']:.4f}) across 5 stratified folds on {len(X):,} samples. "
    f"The runner-up {runner_up['name']} scored {runner_up[METRIC_KEY]:.4f} on {priority_metric}, a gap of {gap_between:.4f}. "
    f"Overfitting gap of {overfitting_gap:.4f} is {'within' if not overfitting_detected else 'above'} the 0.10 threshold, "
    f"indicating {'good' if not overfitting_detected else 'poor'} generalisation. "
    f"{winner['name']} offers the best combination of discriminative power, stability, and generalisation "
    f"for production deployment on this tabular classification task."
)
runner_up_reasoning = (
    f"{runner_up['name']} scored {runner_up['cv_roc_auc_mean']:.4f} AUC "
    f"and is a viable fallback if {winner['name']} deployment infrastructure is unavailable. "
    f"Its overfitting gap of {runner_up['overfitting_gap']:.4f} is "
    f"{'within' if abs(runner_up['overfitting_gap']) <= 0.10 else 'above'} acceptable bounds."
)
genai_narrative = (
    f"{winner['name']} outperformed {len(models_trained) - 1} other models "
    f"with {priority_metric}={winner[METRIC_KEY]:.4f} (AUC={winner['cv_roc_auc_mean']:.4f}). "
    f"Phase 3 stress test: {test_verdict}. "
    f"Higher {priority_metric} means the model better serves the stated business objective."
)

# ══════════════════════════════════════════════════════════════════
# OUTPUT — model_selection.json
# ══════════════════════════════════════════════════════════════════
out = {
    "models_trained":         models_trained,
    "excluded_models":        excluded_models,
    "selected_model":         winner["name"],
    "runner_up_model":        runner_up["name"],
    "selection_reasoning":    selection_reasoning,
    "runner_up_reasoning":    runner_up_reasoning,
    "rejected_models":        rejected_models,
    "class_imbalance_handled": use_balanced,
    "imbalance_strategy":     "class_weight='balanced'" if use_balanced else None,
    "preprocessing_applied":  preprocessing_applied,
    "genai_narrative":        genai_narrative,
    # Phase 3 fields
    "overfitting_detected":   overfitting_detected,
    "overfitting_gap":        round(overfitting_gap, 4),
    "leakage_detected":       leakage_detected,
    "stability_flag":         stability_flag,
    "test_verdict":           test_verdict,
    "test_findings":          test_findings,
    "feature_importance":     feature_importance,
}

REQUIRED_KEYS = [
    "models_trained", "excluded_models", "selected_model", "runner_up_model",
    "selection_reasoning", "runner_up_reasoning", "rejected_models",
    "class_imbalance_handled", "imbalance_strategy", "preprocessing_applied", "genai_narrative",
    "overfitting_detected", "overfitting_gap", "leakage_detected",
    "stability_flag", "test_verdict", "test_findings", "feature_importance",
]
missing = [k for k in REQUIRED_KEYS if k not in out]
if missing:
    raise ValueError(f"SCHEMA VIOLATION — missing keys: {missing}")

pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)

print("\nSCHEMA OK")
print(f"  winner={winner['name']}  auc={winner['cv_roc_auc_mean']:.4f}  std={winner['cv_roc_auc_std']:.4f}")
print(f"  test_verdict={test_verdict}  overfitting={overfitting_detected}  leakage={leakage_detected}")