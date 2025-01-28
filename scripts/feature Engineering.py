import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineering:
    def __init__(self, data_path):
        """
        Initialize the class with the dataset.
        
        Parameters:
        data_path (str): Path to the input dataset file.
        """
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def create_aggregate_features(self):
        """
        Create aggregate features based on customer transactions.
        """
        self.df['Total_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
        self.df['Average_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
        self.df['Transaction_Count'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')
        self.df['Transaction_Amount_STD'] = self.df.groupby('CustomerId')['Amount'].transform('std')

    def extract_date_features(self):
        """
        Extract features from the transaction start time.
        """
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year

    def encode_categorical_features(self):
        """
        Encode categorical variables using Label Encoding and One-Hot Encoding.
        """
        # Label Encoding
        self.df['ProductCategory_LabelEncoded'] = self.label_encoder.fit_transform(self.df['ProductCategory'])

        # One-Hot Encoding
        self.df = pd.get_dummies(self.df, columns=['ChannelId'], prefix='Channel')

    def handle_missing_values(self):
        """
        Handle missing values in the dataset using imputation.
        """
        # Impute missing numerical values with the mean
        num_imputer = SimpleImputer(strategy='mean')
        self.df['Amount'] = num_imputer.fit_transform(self.df[['Amount']])

        # Impute missing categorical values with the mode
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df['ProductCategory'] = cat_imputer.fit_transform(self.df[['ProductCategory']])

    def normalize_and_standardize(self):
        """
        Normalize and standardize numerical features.
        """
        # Standardization
        self.df[['Amount', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Amount_STD']] = \
            self.scaler.fit_transform(self.df[['Amount', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Amount_STD']])

        # Normalization
        self.df[['Transaction_Hour', 'Transaction_Day']] = \
            self.minmax_scaler.fit_transform(self.df[['Transaction_Hour', 'Transaction_Day']])

    def save_processed_data(self, output_path):
        """
        Save the processed dataset to a CSV file.
        
        Parameters:
        output_path (str): Path to save the processed dataset.
        """
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")

    def process(self, output_path):
        """
        Run the entire feature engineering pipeline.
        
        Parameters:
        output_path (str): Path to save the processed dataset.
        """
        self.create_aggregate_features()
        self.extract_date_features()
        self.encode_categorical_features()
        self.handle_missing_values()
        self.normalize_and_standardize()
        self.save_processed_data(output_path)

if __name__ == "__main__":
    feature_engineering = FeatureEngineering(data_path='C:\\Users\\nadew\\10x\\week6\\technical data\\data\\data.csv') 
    feature_engineering.process(output_path='processed_data.csv')
