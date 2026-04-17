import pandas as pd
import json

# Load the dataset
data = pd.read_csv('data/raw/Churn_Modelling.csv')

# Profile of each column
data_profile = {
    'columns': {col: {
        'dtype': str(data[col].dtype),
        'n_unique': data[col].nunique(),
        'missing': data[col].isnull().sum(),
        'example': data[col].iloc[0]
    } for col in data.columns},
    'class_balance': data['Exited'].value_counts(normalize=True).to_dict(),
    'correlations': data.select_dtypes(include=['number']).corr()['Exited'].to_dict(),
    'data_quality_issues': []
}

# Check for potential data quality issues, such as zero balance and no credit cards
zero_balance_no_credit = data[(data['Balance'] == 0) & (data['HasCrCard'] == 0)]
if not zero_balance_no_credit.empty:
    data_profile['data_quality_issues'].append('Customers with zero balance and no credit card found.')

# Check for negative or unrealistic ages (though unlikely in a bank dataset)
negative_ages = data[data['Age'] < 0]
if not negative_ages.empty:
    data_profile['data_quality_issues'].append('Negative ages found in the dataset.')

# Save the findings to JSON
with open('processed/run_b3f5c8bb_data_card.json', 'w') as f:
    def convert(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.float64, np.float32)):
            return float(o)
        raise TypeError

json.dump(data_profile, f, indent=4, default=convert)
