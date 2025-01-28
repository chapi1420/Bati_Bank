import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional

class DefaultEstimatorAndWOE:
    """
    A class to handle default estimation using RFM analysis and WOE binning.
    """
    def __init__(self, data_path: str, target_column: str):
        """
        Initialize the DefaultEstimatorAndWOE class.
        
        Parameters:
        data_path (str): Path to the CSV data file
        target_column (str): Name of the target column
        """
        self.setup_logging()
        self.logger.info("Initializing DefaultEstimatorAndWOE")
        
        try:
            self.df = pd.read_csv(data_path)
            self.target_column = target_column
            self.binned_features = {}
            self.rfm_scores = None
            self.woe_transforms = {}
            
        except Exception as e:
            self.logger.error(f"Error during initialization: {str(e)}")
            raise

    def setup_logging(self):
        """Configure logging for the class"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def calculate_rfm_scores(self) -> None:
        """
        Calculate RFM (Recency, Frequency, Monetary) scores for default estimation.
        """
        try:
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            
            current_date = self.df['TransactionStartTime'].max()
            self.df['Recency'] = self.df.groupby('CustomerId')['TransactionStartTime'].transform(
                lambda x: (current_date - x.max()).days
            )
            
            self.df['Frequency'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')
            
            self.df['Monetary'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
            
            self.df['R_Score'] = self.scale_rfm_scores(self.df['Recency'], reverse=True)
            self.df['F_Score'] = self.scale_rfm_scores(self.df['Frequency'])
            self.df['M_Score'] = self.scale_rfm_scores(self.df['Monetary'])
            
            self.df['RFM_Score'] = (self.df['R_Score'] + self.df['F_Score'] + self.df['M_Score']) / 3
            
            self.logger.info("RFM scores calculated successfully")
            
        except Exception as e:
            self.logger.error(f"Error in RFM calculation: {str(e)}")
            raise

    @staticmethod
    def scale_rfm_scores(series: pd.Series, reverse: bool = False) -> pd.Series:
        """
        Scale RFM components to 0-1 range.
        
        Parameters:
        series (pd.Series): Series to scale
        reverse (bool): If True, reverse the scoring (for Recency)
        
        Returns:
        pd.Series: Scaled series
        """
        if reverse:
            return (series.max() - series) / (series.max() - series.min())
        return (series - series.min()) / (series.max() - series.min())

    def assign_default_labels(self, threshold: float = 0.5) -> None:
        """
        Assign default labels based on RFM scores.
        
        Parameters:
        threshold (float): Threshold for good/bad classification
        """
        try:
            if self.df is None or 'RFM_Score' not in self.df.columns:
                raise ValueError("RFM scores must be calculated first")
            
            self.df['Default_Label'] = np.where(self.df['RFM_Score'] >= threshold, 'Good', 'Bad')
            self.df[self.target_column] = (self.df['Default_Label'] == 'Bad').astype(int)
            
            label_dist = self.df['Default_Label'].value_counts()
            self.logger.info(f"Default labels distribution: {label_dist.to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error in default label assignment: {str(e)}")
            raise

    def perform_woe_binning(self, feature: str, n_bins: int = 10) -> Bunch:
        """
        Perform Weight of Evidence binning on a feature.
        
        Parameters:
        feature (str): Feature to bin
        n_bins (int): Number of bins to create
        
        Returns:
        Bunch: WOE and IV values
        """
        try:
            if self.target_column not in self.df.columns:
                raise ValueError("Target column not found. Run assign_default_labels first.")

            self.df[f'{feature}_bin'] = pd.qcut(
                self.df[feature], 
                q=n_bins, 
                duplicates='drop'
            )
            
            woe_data = self._calculate_woe_iv(feature)
            
            self.binned_features[feature] = woe_data
            
            self.woe_transforms[feature] = dict(zip(woe_data.index, woe_data['WOE']))
            
            self.logger.info(f"WOE binning completed for feature: {feature}")
            return Bunch(
                woe=woe_data['WOE'],
                iv=woe_data['IV'].sum(),
                bins=woe_data.index
            )
            
        except Exception as e:
            self.logger.error(f"Error in WOE binning: {str(e)}")
            raise

    def _calculate_woe_iv(self, feature: str) -> pd.DataFrame:
        """
        Calculate Weight of Evidence and Information Value for a binned feature.
        
        Parameters:
        feature (str): The binned feature name
        
        Returns:
        pd.DataFrame: DataFrame containing WOE and IV calculations
        """
        try:
            grouped = self.df.groupby(f'{feature}_bin')[self.target_column].agg(['count', 'sum'])
            
            grouped['non_events'] = grouped['count'] - grouped['sum']
            grouped['events'] = grouped['sum']
            
            grouped['dist_non_events'] = grouped['non_events'] / grouped['non_events'].sum()
            grouped['dist_events'] = grouped['events'] / grouped['events'].sum()
            
            grouped['WOE'] = np.log(grouped['dist_non_events'] / grouped['dist_events'])
            grouped['IV'] = (grouped['dist_non_events'] - grouped['dist_events']) * grouped['WOE']
            
            return grouped
            
        except Exception as e:
            self.logger.error(f"Error in WOE/IV calculation: {str(e)}")
            raise

    def transform_feature_woe(self, feature: str, data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Transform a feature using stored WOE values.
        
        Parameters:
        feature (str): Feature to transform
        data (pd.DataFrame, optional): Data to transform, defaults to stored DataFrame
        
        Returns:
        pd.Series: WOE-transformed feature
        """
        try:
            if data is None:
                data = self.df
                
            if feature not in self.woe_transforms:
                raise ValueError(f"No WOE transformation found for feature: {feature}")
                
            bins = pd.qcut(data[feature], q=len(self.woe_transforms[feature]), duplicates='drop')
            
            woe_values = bins.map(self.woe_transforms[feature])
            
            return woe_values
            
        except Exception as e:
            self.logger.error(f"Error in WOE transformation: {str(e)}")
            raise

    def plot_woe(self, feature: str) -> None:
        """
        Plot WOE values for a binned feature.
        
        Parameters:
        feature (str): Feature to plot
        """
        try:
            if feature not in self.binned_features:
                raise ValueError(f"No WOE data found for feature: {feature}")
            
            woe_data = self.binned_features[feature]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(woe_data['WOE'])), woe_data['WOE'])
            plt.title(f'Weight of Evidence (WOE) for {feature}')
            plt.xlabel('Bins')
            plt.ylabel('WOE')
            plt.xticks(range(len(woe_data['WOE'])), 
                      [str(bin_) for bin_ in woe_data.index], 
                      rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error in WOE plotting: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance based on Information Value.
        
        Returns:
        pd.Series: Series of features and their IV values
        """
        try:
            feature_iv = {}
            for feature in self.binned_features:
                feature_iv[feature] = self.binned_features[feature]['IV'].sum()
            
            return pd.Series(feature_iv).sort_values(ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error in getting feature importance: {str(e)}")
            raise

    def save_woe_transforms(self, path: str) -> None:
        """
        Save WOE transformations to file.
        
        Parameters:
        path (str): Path to save the transformations
        """
        try:
            pd.to_pickle(self.woe_transforms, path)
            self.logger.info(f"WOE transformations saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving WOE transforms: {str(e)}")
            raise

    def load_woe_transforms(self, path: str) -> None:
        """
        Load WOE transformations from file.
        
        Parameters:
        path (str): Path to load the transformations from
        """
        try:
            self.woe_transforms = pd.read_pickle(path)
            self.logger.info(f"WOE transformations loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading WOE transforms: {str(e)}")
            raise


def main():
    estimator = DefaultEstimatorAndWOE(
        data_path='C:\\Users\\nadew\\10x\\week6\\technical data\\data\\featured_data.csv',
        target_column='FraudResult'
    )
    
    estimator.calculate_rfm_scores()
    
    estimator.assign_default_labels(threshold=0.6)
    
    numeric_features = ['Amount', 'Frequency', 'Recency', 'Monetary']
    
    for feature in numeric_features:
        woe_results = estimator.perform_woe_binning(feature)
        
        estimator.plot_woe(feature)
        
        print(f"\nFeature: {feature}")
        print(f"Information Value: {woe_results.iv:.4f}")
    
    feature_importance = estimator.get_feature_importance()
    print("\nFeature Importance (IV):")
    print(feature_importance)
    
    estimator.save_woe_transforms('woe_transforms.pkl')

if __name__ == "__main__":
    main()