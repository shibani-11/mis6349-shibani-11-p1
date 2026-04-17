import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # Load the dataset
    df = pd.read_csv('data/raw/Churn_Modelling.csv')
    
    # Display basic information about the dataset
    print("Basic Information:")
    print(df.info())
    
    # Display statistics
    print("\nStatistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Compute class balance for 'Exited'
    print("\nClass Balance (Exited):")
    print(df['Exited'].value_counts(normalize=True))
    
    # Correlation analysis
    print("\nCorrelation Matrix:")
    corr_matrix = df.drop(columns=['Surname', 'Geography', 'Gender']).corr()
    print(corr_matrix)
    
    # Visualize correlation matrix
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == '__main__':
    main()