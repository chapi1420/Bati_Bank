import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[
        logging.FileHandler('eda.log'),  # Log to a file
        logging.StreamHandler()  # Log to the console
    ]
)

class EDA:
    def __init__(self, data_path):
        """
        Initialize the EDA class with the path to the dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file (e.g., CSV).
        """
        self.data = pd.read_csv(data_path)
        logging.info(f"Dataset loaded from {data_path}")

    def overview(self):
        """
        Provide an overview of the dataset, including the number of rows, columns, and data types.
        """
        logging.info("Starting dataset overview...")
        print("Dataset Overview:")
        print(f"Number of Rows: {self.data.shape[0]}")
        print(f"Number of Columns: {self.data.shape[1]}")
        print("\nData Types:")
        print(self.data.dtypes)
        print("\nFirst 5 Rows:")
        print(self.data.head())
        logging.info("Dataset overview completed.")

    def summary_statistics(self):
        """
        Calculate and display summary statistics for the dataset.
        """
        logging.info("Calculating summary statistics...")
        print("\nSummary Statistics:")
        print(self.data.describe(include='all'))
        logging.info("Summary statistics calculation completed.")

    def numerical_distribution(self):
        """
        Visualize the distribution of numerical features using histograms.
        """
        logging.info("Visualizing numerical feature distributions...")
        numerical_columns = ['Amount', 'Value']  
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(numerical_columns, 1):
            plt.subplot(2, 2, i)
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()
        logging.info("Numerical feature distributions visualized.")

    def categorical_distribution(self):
        """
        Analyze the distribution of categorical features using bar plots.
        """
        logging.info("Visualizing categorical feature distributions...")
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId', 'FraudResult']  # Focus on categorical columns
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(categorical_columns, 1):
            plt.subplot(3, 2, i)
            sns.countplot(y=self.data[column], order=self.data[column].value_counts().index)
            plt.title(f'Distribution of {column}')
        plt.tight_layout()
        plt.show()
        logging.info("Categorical feature distributions visualized.")

    def correlation_analysis(self):
        """
        Perform correlation analysis on numerical features and visualize the correlation matrix.
        """
        logging.info("Performing correlation analysis...")
        numerical_columns = ['Amount', 'Value'] 
        correlation_matrix = self.data[numerical_columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.show()
        logging.info("Correlation analysis completed.")

    def missing_values(self):
        """
        Identify and display missing values in the dataset.
        """
        logging.info("Checking for missing values...")
        missing_values = self.data.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if len(missing_values) > 0:
            print("\nMissing Values:")
            print(missing_values)
            logging.warning(f"Missing values found: {missing_values.to_dict()}")
        else:
            print("\nNo Missing Values Found.")
            logging.info("No missing values found.")

    def outlier_detection(self):
        """
        Detect outliers in numerical features using box plots.
        """
        logging.info("Detecting outliers in numerical features...")
        numerical_columns = ['Amount', 'Value']  
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(numerical_columns, 1):
            plt.subplot(2, 2, i)
            sns.boxplot(x=self.data[column])
            plt.title(f'Box Plot of {column}')
        plt.tight_layout()
        plt.show()
        logging.info("Outlier detection completed.")

    def run_all(self):
        """
        Run all EDA tasks sequentially.
        """
        logging.info("Starting EDA process...")
        self.overview()
        self.summary_statistics()
        self.numerical_distribution()
        self.categorical_distribution()
        self.correlation_analysis()
        self.missing_values()
        self.outlier_detection()
        logging.info("EDA process completed.")

