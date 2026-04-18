# Recommendations based on model evaluations
recommendations = {
    'Best Model': 'Gradient Boosting',
    'Recommendation': 'Utilize Gradient Boosting for churn prediction due to its highest AUC score. Random Forest is also a reliable option, but explore further tuning for better results.'
}

# Save recommendations to JSON file
import json
with open('processed/run_6337fd0e_recommendation.json', 'w') as json_file:
    json.dump(recommendations, json_file)
