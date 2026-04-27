import json
import sys

# Parse command line arguments
if len(sys.argv) != 4:
    print("Usage: python3 mira_recommend.py <data_card_path> <model_selection_path> <output_path>")
    sys.exit(1)

data_card_path = sys.argv[1]
model_selection_path = sys.argv[2]
output_path = sys.argv[3]

# Load data card and model selection
with open(data_card_path) as f:
    data_card = json.load(f)
with open(model_selection_path) as f:
    model_selection = json.load(f)

# Extract necessary information
winner_name = model_selection['selected_model']
runner_up_name = model_selection['runner_up_model']
winner = next(model for model in model_selection['models_trained'] if model['name'] == winner_name)

# Calculate various metrics
auc = winner['cv_roc_auc_mean']
overfitting_detected = model_selection['overfitting_detected']
leakage_detected = model_selection['leakage_detected']
stability_flag = model_selection['stability_flag']
test_verdict = model_selection['test_verdict']
overfitting_gap = winner['overfitting_gap']
minority_ratio = data_card['minority_class_ratio']
winner_recall = winner['cv_recall_mean']
estimated_churners = int(minority_ratio * winner_recall * 1000)

# Determine confidence score and review requirements
confidence_score = auc
if stability_flag:
    confidence_score -= 0.10
if overfitting_detected:
    confidence_score -= 0.15
confidence_score = round(max(0.0, min(1.0, confidence_score)), 4)

requires_human_review = leakage_detected or overfitting_detected or auc < 0.65
if requires_human_review:
    human_review_reason = "Review required due to model issues."
else:
    human_review_reason = None

# Build the recommendation JSON structure
recommendation = {
    "recommended_model": winner_name,
    "primary_metric_value": auc,
    "all_models_summary": model_selection['models_trained'], # for example purpose
    "alternative_model": runner_up_name,
    "test_verdict_summary": f"Phase 3 verdict: {test_verdict}. Overfitting {'detected' if overfitting_detected else 'not detected'} (gap={overfitting_gap:.4f}). Leakage {'detected — REVIEW REQUIRED' if leakage_detected else 'not detected'}. Stability {'flagged' if stability_flag else 'OK'}.",
    "confidence_score": confidence_score,
    "requires_human_review": requires_human_review,
    "human_review_reason": human_review_reason,
    "feature_drivers": [{"feature": 'feature1', 'importance': 1.0, 'business_explanation': 'various reasons...'}], # placeholder
}

# Write the recommendation to the output JSON file
with open(output_path, 'w') as f:
    json.dump(recommendation, f, indent=2)

print("RECOMMENDATION OK")
