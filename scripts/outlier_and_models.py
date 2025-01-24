import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class OutlierAnalysisAndModeling:
    def __init__(self, data_path):
        """
        Initialize the class with the dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the dataset file (e.g., CSV).
        """
        self.data = pd.read_csv(data_path)
        self.transformed_data = None

    def investigate_outliers(self, column):
        """
        Investigate outliers in a specific column using the IQR method.
        
        Parameters:
        -----------
        column : str
            The column to analyze for outliers.
        
        Returns:
        --------
        outliers : DataFrame
            Rows in the dataset that are considered outliers for the given column.
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        print(f"Outliers in {column}: {len(outliers)}")
        return outliers

    def apply_log_transformation(self, column):
        """
        Apply log transformation to a column to reduce skewness and the impact of outliers.
        
        Parameters:
        -----------
        column : str
            The column to apply the log transformation to.
        """
        self.data[column + '_log'] = np.log1p(self.data[column])  
    def visualize_transformed_distributions(self):
        """
        Visualize the distributions of log-transformed columns.
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(self.data['Amount_log'], kde=True)
        plt.title('Log Transformed Amount')

        plt.subplot(1, 2, 2)
        sns.histplot(self.data['Value_log'], kde=True)
        plt.title('Log Transformed Value')

        plt.tight_layout()
        plt.show()

    def train_robust_model(self, target_column):
        """
        Train a robust model (Random Forest) using log-transformed features.
        
        Parameters:
        -----------
        target_column : str
            The target column for the model (e.g., 'FraudResult').
        """
        # Prepare data for modeling
        X = self.data[['Amount_log', 'Value_log']]  
        y = self.data[target_column]  

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))

        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        print("\nFeature Importance:")
        print(feature_importance)

    def run_all(self):
        """
        Run all steps sequentially: outlier investigation, data transformation, and modeling.
        """
        print("Investigating Outliers...")
        amount_outliers = self.investigate_outliers('Amount')
        value_outliers = self.investigate_outliers('Value')

        print("\nOutliers in Amount:")
        print(amount_outliers.head())

        print("\nOutliers in Value:")
        print(value_outliers.head())

        print("\nApplying Log Transformations...")
        self.apply_log_transformation('Amount')
        self.apply_log_transformation('Value')

        print("\nVisualizing Transformed Distributions...")
        self.visualize_transformed_distributions()

        print("\nTraining Robust Model...")
        self.train_robust_model('FraudResult')
