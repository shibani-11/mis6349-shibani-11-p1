import pandas as pd
import json

# Read the full dataset
data_path = 'data/raw/train.csv'
df = pd.read_csv(data_path)

# Initialize a dictionary to store the findings
report = {}

# 1. Basic dataset statistics
report['row_count'] = df.shape[0]
report['column_count'] = df.shape[1]
report['inferred_task_type'] = 'classification'

# 2. Column profiles
columns = []
numeric_columns = []
categorical_columns = []
id_columns = []
text_columns = []

for column in df.columns:
    col_profile = {}
    col_profile['name'] = column
    col_profile['dtype'] = str(df[column].dtype.name)
    col_profile['null_count'] = df[column].isnull().sum()
    col_profile['null_percentage'] = df[column].isnull().mean() * 100
    col_profile['unique_value_count'] = df[column].nunique()
    col_profile['sample_values'] = df[column].dropna().unique()[:3].tolist()

    # Determine the column type
    if pd.api.types.is_numeric_dtype(df[column]):
        col_profile['type'] = 'numeric'
        numeric_columns.append(column)
    elif pd.api.types.is_categorical_dtype(df[column]):
        col_profile['type'] = 'categorical'
        categorical_columns.append(column)
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        col_profile['type'] = 'datetime'
    elif col_profile['unique_value_count'] == df.shape[0]:
        col_profile['type'] = 'id'
        id_columns.append(column)
    else:
        col_profile['type'] = 'text'
        text_columns.append(column)

    columns.append(col_profile)

report['columns'] = columns
report['numeric_columns'] = numeric_columns
report['categorical_columns'] = categorical_columns
report['id_columns'] = id_columns
report['text_columns'] = text_columns

# 3. Analyze target column distribution
target_counts = df['Loan Status'].value_counts().to_frame().reset_index().to_dict(orient='split')['data']
minority_class_count = min(tc[1] for tc in target_counts)
majority_class_count = max(tc[1] for tc in target_counts)
report['target_distribution'] = target_counts
report['imbalance_ratio'] = minority_class_count / majority_class_count
report['class_imbalance_detected'] = minority_class_count < (0.2 * sum(tc[1] for tc in target_counts))

# 4. Data quality issues
overall_missing_count = df.isnull().sum().sum()
report['overall_missing_pct'] = (overall_missing_count / df.size) * 100
report['duplicate_row_count'] = df.duplicated().sum()

quality_issues = []
if report['duplicate_row_count'] > 0:
    quality_issues.append('Duplicate rows found')
for col in df.columns:
    if df[col].nunique() == 1:
        quality_issues.append(f'Column {col} has only one unique value')
report['quality_issues'] = quality_issues

# 5. Statistics for numeric columns
numeric_stats = {}
for col in numeric_columns:
    numeric_stats[col] = {
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max(),
        '25%': df[col].quantile(0.25),
        '50%': df[col].quantile(0.5),
        '75%': df[col].quantile(0.75)
    }
report['numeric_stats'] = numeric_stats

# 6. GenAI narrative
genai_narrative = (
    "The dataset provides insights into potential loan applicants' risk profiles, "
    "helping to identify high-risk individuals and support smarter credit approval decisions. "
    "By analyzing factors contributing to defaults, we can enhance our risk management strategies."
)
report['genai_narrative'] = genai_narrative

# Save the findings as JSON
output_path = 'processed/run_c4d60dd9_data_exploration.json'
with open(output_path, 'w') as f:
    json.dump(report, f, indent=4, default=str)