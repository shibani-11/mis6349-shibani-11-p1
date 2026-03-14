import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import json
import time

# Load dataset
train = pd.read_csv("train.csv")

# Target
y = train["Loan Status"]

# Features
X = train.drop(columns=["Loan Status", "ID"])

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1]

    latency = (time.time() - start) * 1000

    metrics = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds)),
        "recall": float(recall_score(y_val, preds)),
        "f1": float(f1_score(y_val, preds)),
        "roc_auc": float(roc_auc_score(y_val, probs))
    }

    results.append({
        "name": name,
        "metrics": metrics,
        "latency_ms": round(latency,2),
        "training_time_sec": round(latency/1000,2),
        "inference_cost": "low",
        "interpretability": "high" if name=="LogisticRegression" else "medium",
        "robustness_notes": "baseline",
        "deployment_notes": "ready"
    })


agent_input = {
    "dataset_name": "Loan Default Prediction",
    "problem_type": "classification",
    "business_context": "Predict loan default risk for lending decisions",
    "models": results
}

with open("data/loan_model_results.json","w") as f:
    json.dump(agent_input,f,indent=2)

print("Generated agent input file.")
