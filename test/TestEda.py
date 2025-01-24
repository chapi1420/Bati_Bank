import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TestEDA:
    def __init__(self):
        """
        Initialize the TestEDA class with a synthetic dataset.
        """
        self.data = self.create_synthetic_data()

    def create_synthetic_data(self):
        """
        Create a synthetic dataset for testing the EDA class.
        """
        np.random.seed(42)  # For reproducibility
        n_rows = 100  # Number of rows in the synthetic dataset

        data = {
            "TransactionId": np.arange(1, n_rows + 1),
            "BatchId": np.random.randint(1, 10, size=n_rows),
            "AccountId": np.random.randint(1000, 2000, size=n_rows),
            "SubscriptionId": np.random.randint(1, 5, size=n_rows),
            "CustomerId": np.random.randint(5000, 6000, size=n_rows),
            "CurrencyCode": np.random.choice(["USD", "EUR", "GBP"], size=n_rows),
            "CountryCode": np.random.randint(1, 10, size=n_rows),
            "ProviderId": np.random.randint(1, 5, size=n_rows),
            "ProductId": np.random.choice(["P1", "P2", "P3"], size=n_rows),
            "ProductCategory": np.random.choice(["Electronics", "Clothing", "Groceries"], size=n_rows),
            "ChannelId": np.random.choice(["Web", "Android", "iOS"], size=n_rows),
            "Amount": np.random.uniform(-1000, 1000, size=n_rows),
            "Value": np.abs(np.random.uniform(0, 1000, size=n_rows)),
            "TransactionStartTime": pd.date_range(start="2023-01-01", periods=n_rows, freq="H"),
            "PricingStrategy": np.random.choice(["A", "B", "C"], size=n_rows),
            "FraudResult": np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1]),  # 10% fraud
        }

        return pd.DataFrame(data)

    def test_overview(self, eda):
        """
        Test the overview functionality of the EDA class.
        """
        print("Testing Overview...")
        eda.overview()

    def test_summary_statistics(self, eda):
        """
        Test the summary statistics functionality of the EDA class.
        """
        print("\nTesting Summary Statistics...")
        eda.summary_statistics()

    def test_numerical_distribution(self, eda):
        """
        Test the numerical distribution visualization functionality of the EDA class.
        """
        print("\nTesting Numerical Distribution...")
        eda.numerical_distribution()

    def test_categorical_distribution(self, eda):
        """
        Test the categorical distribution visualization functionality of the EDA class.
        """
        print("\nTesting Categorical Distribution...")
        eda.categorical_distribution()

    def test_correlation_analysis(self, eda):
        """
        Test the correlation analysis functionality of the EDA class.
        """
        print("\nTesting Correlation Analysis...")
        eda.correlation_analysis()

    def test_missing_values(self, eda):
        """
        Test the missing values functionality of the EDA class.
        """
        print("\nTesting Missing Values...")
        eda.missing_values()

    def test_outlier_detection(self, eda):
        """
        Test the outlier detection functionality of the EDA class.
        """
        print("\nTesting Outlier Detection...")
        eda.outlier_detection()

    def run_all_tests(self, eda):
        """
        Run all tests sequentially.
        """
        self.test_overview(eda)
        self.test_summary_statistics(eda)
        self.test_numerical_distribution(eda)
        self.test_categorical_distribution(eda)
        self.test_correlation_analysis(eda)
        self.test_missing_values(eda)
        self.test_outlier_detection(eda)


# if __name__ == "__main__":
#     tester = TestEDA()

#     from eda import EDA  
#     eda = EDA('synthetic_data.csv')  
#     # Run all tests
#     tester.run_all_tests(eda)