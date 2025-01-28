import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging

class CreditScoringModel:
    def __init__(self, model_type='rf'):
        """
        Initialize the Credit Scoring Model.
        
        Parameters:
        model_type (str): Type of model to use ('rf' for Random Forest or 'lr' for Logistic Regression)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the model"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare and split the data for training.
        
        Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        test_size (float): Proportion of dataset to include in the test split
        random_state (int): Random state for reproducibility
        
        Returns:
        tuple: Training and testing sets
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.logger.info("Data preparation completed successfully")
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise

    def build_model(self):
        """
        Build the selected model with predefined parameters.
        """
        try:
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(random_state=42)
                self.param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'lr':
                self.model = LogisticRegression(random_state=42)
                self.param_grid = {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            else:
                raise ValueError("Unsupported model type")
            
            self.logger.info(f"Model {self.model_type} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise

    def train_model(self, X_train, y_train, cv=5):
        """
        Train the model using GridSearchCV for hyperparameter tuning.
        
        Parameters:
        X_train: Training features
        y_train: Training target
        cv (int): Number of cross-validation folds
        """
        try:
            grid_search = GridSearchCV(
                self.model,
                self.param_grid,
                cv=cv,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model using various metrics.
        
        Parameters:
        X_test: Test features
        y_test: Test target
        
        Returns:
        dict: Dictionary containing various evaluation metrics
        """
        try:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error in model evaluation: {str(e)}")
            raise

    def save_model(self, model_path, scaler_path):
        """
        Save the trained model and scaler to disk.
        
        Parameters:
        model_path (str): Path to save the model
        scaler_path (str): Path to save the scaler
        """
        try:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Model saved to {model_path}")
            self.logger.info(f"Scaler saved to {scaler_path}")
        
        except Exception as e:
            self.logger.error(f"Error in saving model: {str(e)}")
            raise

    def load_model(self, model_path, scaler_path):
        """
        Load a trained model and scaler from disk.
        
        Parameters:
        model_path (str): Path to the saved model
        scaler_path (str): Path to the saved scaler
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.logger.info("Model and scaler loaded successfully")
        
        except Exception as e:
            self.logger.error(f"Error in loading model: {str(e)}")
            raise

    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        X: Features to predict on
        
        Returns:
        tuple: Predictions and prediction probabilities
        """
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            return predictions, probabilities
        
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise