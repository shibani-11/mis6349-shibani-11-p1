import pandas as pd

# Load the dataset
file_path = 'data/raw/Churn_Modelling.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print("DataFrame Head:")
print(df.head())

# Profile each column
data_profile = df.describe(include='all')

# Check class balance for 'Exited' column
class_balance = df['Exited'].value_counts(normalize=True)

# Check for missing values
missing_values = df.isnull().sum()

# Check correlations
correlation_matrix = df.select_dtypes(include='number').corr()

# Output results to JSON
output = {
    'data_profile': data_profile.to_dict(),
    'class_balance': class_balance.to_dict(),
    'missing_values': missing_values.to_dict(),
    'correlation_matrix': correlation_matrix.to_dict()
}

# Save output to JSON
import json
with open('processed/run_6337fd0e_data_card.json', 'w') as json_file:
    json.dump(output, json_file)
