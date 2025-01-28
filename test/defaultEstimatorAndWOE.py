import unittest
import pandas as pd
import numpy as np
import tempfile
import os

class TestDefaultEstimatorAndWOE(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file with test data
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create sample test data
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'CustomerId': range(1, n_samples + 1),
            'TransactionId': range(1, n_samples + 1),
            'TransactionStartTime': pd.date_range(start='2023-01-01', periods=n_samples),
            'Amount': np.random.uniform(100, 1000, n_samples),
            'Feature1': np.random.normal(0, 1, n_samples),
            'Target': np.random.binomial(1, 0.3, n_samples)
        }
        
        self.test_df = pd.DataFrame(data)
        self.test_df.to_csv(self.data_path, index=False)
        
        # Initialize the class
        self.estimator = DefaultEstimatorAndWOE(self.data_path, 'Target')

    def tearDown(self):
        # Clean up temporary files
        os.remove(self.data_path)
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test if the class initializes correctly"""
        self.assertIsInstance(self.estimator.df, pd.DataFrame)
        self.assertEqual(self.estimator.target_column, 'Target')
        self.assertEqual(len(self.estimator.binned_features), 0)

    def test_calculate_rfm_scores(self):
        """Test RFM score calculation"""
        self.estimator.calculate_rfm_scores()
        
        # Check if RFM columns were created
        self.assertTrue('Recency' in self.estimator.df.columns)
        self.assertTrue('Frequency' in self.estimator.df.columns)
        self.assertTrue('Monetary' in self.estimator.df.columns)
        self.assertTrue('RFM_Score' in self.estimator.df.columns)
        
        # Check if RFM scores are within expected range
        self.assertTrue(self.estimator.df['RFM_Score'].min() >= 0)
        self.assertTrue(self.estimator.df['RFM_Score'].max() <= 3)

    def test_assign_default_labels(self):
        """Test default label assignment"""
        self.estimator.calculate_rfm_scores()
        self.estimator.assign_default_labels(threshold=0.5)
        
        # Check if Default_Label column was created
        self.assertTrue('Default_Label' in self.estimator.df.columns)
        
        # Check if labels are either 'Good' or 'Bad'
        unique_labels = self.estimator.df['Default_Label'].unique()
        self.assertEqual(set(unique_labels), {'Good', 'Bad'})

    def test_perform_woe_binning(self):
        """Test WOE binning"""
        # First calculate RFM scores and assign labels
        self.estimator.calculate_rfm_scores()
        self.estimator.assign_default_labels()
        
        # Perform WOE binning on Feature1
        result = self.estimator.perform_woe_binning('Feature1')
        
        # Check if results are returned correctly
        self.assertTrue(hasattr(result, 'woe'))
        self.assertTrue(hasattr(result, 'iv'))
        
        # Check if binned feature was stored
        self.assertTrue('Feature1' in self.estimator.binned_features)
        
        # Check if WOE values are finite
        self.assertTrue(np.all(np.isfinite(result.woe)))

    def test_plot_woe(self):
        """Test WOE plotting functionality"""
        # First perform necessary calculations
        self.estimator.calculate_rfm_scores()
        self.estimator.assign_default_labels()
        self.estimator.perform_woe_binning('Feature1')
        
        # Test plotting (this just checks if the method runs without errors)
        try:
            self.estimator.plot_woe('Feature1')
            plot_success = True
        except Exception as e:
            plot_success = False
        
        self.assertTrue(plot_success)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)