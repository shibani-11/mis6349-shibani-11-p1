import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the data
data = pd.read_csv('data/raw/Churn_Modelling.csv')

# Initial data inspection
column_summary = data.describe(include='all')

# Check for missing values
data_quality_issues = data.isnull().sum()

# Class balance for 'Exited'
class_balance = data['Exited'].value_counts(normalize=True) * 100

# Correlation matrix
correlation_matrix = data.corr()

# Identify highly correlated features
highly_correlated = [(column1, column2, pearsonr(data[column1], data[column2])[0])
                     for column1 in data.select_dtypes(include=[np.number]).columns
                     for column2 in data.select_dtypes(include=[np.number]).columns 
                     if column1 != column2 and pearsonr(data[column1], data[column2])[0] > 0.75]

# Convert results to JSON serializable structures
column_summary_dict = column_summary.to_dict()
data_quality_issues_dict = data_quality_issues.to_dict()
class_balance_dict = class_balance.to_dict()
correlation_matrix_dict = correlation_matrix.to_dict()
highly_correlated_dict = [{'column1': tup[0], 'column2': tup[1], 'correlation': tup[2]} for tup in highly_correlated]

# Write findings to JSON file
import json
findings = {
    'column_summary': column_summary_dict,
    'data_quality_issues':
    data_quality_issues_dict,
    'class_balance': class_balance_dict,
    'correlation_matrix': correlation_matrix_dict,
    'highly_correlated': highly_correlated_dict
}

with open('processed/run_6a1cde32_data_card.json', 'w') as f:
    json.dump(findings, f, indent=4)