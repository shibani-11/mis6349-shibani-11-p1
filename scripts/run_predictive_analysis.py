import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('data/raw/train.csv', usecols=lambda column: column != 'ID')

# Split the data into features and target
X = data.drop(columns='Loan Status')
y = data['Loan Status']

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(exclude=['object']).columns

# Address class imbalance (log entry)
# Note: We will likely implement SMOTE or similar balancing techniques in model training phase

# Handle categorical data
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Handle skewed features
skewed_features = [  # Add the skewed column names
    'Loan Amount', 'Interest Rate'
]
for feature in skewed_features:
    X[feature] = X[feature].apply(lambda x: np.log1p(x) if x > 0 else 0)

# Split the data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numeric features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

# Save preprocessed data for further analysis (temporary)
X_train.to_csv('processed/X_train_preprocessed.csv', index=False)
X_val.to_csv('processed/X_val_preprocessed.csv', index=False)
