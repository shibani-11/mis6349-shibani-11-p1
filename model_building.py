import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import json
import time

# Load dataset
file_path = 'data/raw/train.csv'
data = pd.read_csv(file_path, skipinitialspace=True)
print('Columns:', data.columns)  # Debug: List all columns to verify

# Drop uninformative columns
columns_to_drop = ['ID', 'Payment Plan']
data = data.drop(columns=columns_to_drop)

# Handle missing values
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = data.select_dtypes(include=['str']).columns

# Preprocessing for numerical data: impute missing values, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing, then encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder())
])

# Create column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Split data
X = data.drop(columns=['Loan Status'])
y = data['Loan Status']

preprocessed_X = preprocessor.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(preprocessed_X, y, test_size=0.2, random_state=42)

# Models to consider
models = [
    ('Logistic Regression', LogisticRegression(class_weight='balanced', max_iter=1000)),
    ('Random Forest', RandomForestClassifier(class_weight='balanced', n_jobs=-1)),
    ('XGBoost', XGBClassifier(scale_pos_weight=99)),
    ('LightGBM', LGBMClassifier(class_weight='balanced'))
]

# List to store model evaluations
evaluated_models = []
start_time = time.time()

for name, model in models:
    # Time the training process
    train_start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

    # Validation predictions
    y_pred = model.predict(X_val)

    
    # Calculate scores
    model_metrics = {
        'model_name': name,
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred),
        'training_time_seconds': train_time,
        'cross_val_score_mean': cv_scores.mean(),
        'cross_val_score_std': cv_scores.std()
    }

    evaluated_models.append(model_metrics)

# Time elapsed
end_time = time.time()
total_time = end_time - start_time

# Save results
results = {
    "models_considered": [name for (name, _) in models],
    "models_evaluated": evaluated_models,
    "models_skipped": {},
    "primary_metric": "roc_auc",
    "best_model_by_metric": max(evaluated_models, key=lambda x: x['roc_auc'])['model_name']
}

output_path = 'processed/run_c4d60dd9_model_building.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)
