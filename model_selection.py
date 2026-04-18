import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json

# Load the dataset
file_path = 'data/raw/Churn_Modelling.csv'
df = pd.read_csv(file_path)

# Prepare the data
X = df.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname'])  # drop non-feature columns
X = pd.get_dummies(X)

# Target variable
y = df['Exited']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
evaluated_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True)
}

# Dictionary to hold the evaluation results
model_performance = {}

# Train and evaluate each model
for model_name, model in evaluated_models.items():
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    model_performance[model_name] = roc_auc

# Output results to JSON
with open('processed/run_6337fd0e_model_selection.json', 'w') as json_file:
    json.dump(model_performance, json_file)
