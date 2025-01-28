from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import pandas as pd
from typing import List, Optional
import uvicorn
import logging
import train_model

class CreditScoringAPI:
    def __init__(self, model_instance):
        """
        Initialize the API with a trained model instance.
        
        Parameters:
        model_instance: Trained CreditScoringModel instance
        """
        self.app = FastAPI(title="Credit Scoring API",
                          description="API for credit risk assessment")
        self.model = model_instance
        self.setup_logging()
        self.setup_routes()

    def setup_logging(self):
        """Configure logging for the API"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_routes(self):
        """Set up API routes and endpoints"""
        
        class PredictionRequest(BaseModel):
            features: List[float]
            customer_id: Optional[str] = None

        class PredictionResponse(BaseModel):
            customer_id: Optional[str]
            prediction: int
            probability: float
            risk_level: str
            recommended_action: str

        @self.app.get("/")
        def read_root():
            return {"message": "Welcome to the Credit Scoring API"}

        @self.app.post("/predict", response_model=PredictionResponse)
        def predict(request: PredictionRequest):
            try:
                # Convert features to numpy array
                features = np.array(request.features).reshape(1, -1)
                
                # Make prediction
                prediction, probability = self.model.predict(features)
                
                # Determine risk level and recommended action
                risk_level = self.determine_risk_level(probability[0])
                recommended_action = self.get_recommendation(risk_level)
                
                return PredictionResponse(
                    customer_id=request.customer_id,
                    prediction=int(prediction[0]),
                    probability=float(probability[0]),
                    risk_level=risk_level,
                    recommended_action=recommended_action
                )
            
            except Exception as e:
                self.logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/model-info")
        def get_model_info():
            """Return information about the model"""
            try:
                return {
                    "model_type": self.model.model_type,
                    "model_parameters": str(self.model.model.get_params()),
                    "feature_count": len(self.model.model.feature_importances_) if hasattr(self.model.model, 'feature_importances_') else "N/A"
                }
            except Exception as e:
                self.logger.error(f"Error retrieving model info: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

    def determine_risk_level(self, probability):
        """
        Determine risk level based on probability.
        
        Parameters:
        probability (float): Prediction probability
        
        Returns:
        str: Risk level category
        """
        if probability < 0.2:
            return "Very Low Risk"
        elif probability < 0.4:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"

    def get_recommendation(self, risk_level):
        """
        Get recommendation based on risk level.
        
        Parameters:
        risk_level (str): Risk level category
        
        Returns:
        str: Recommended action
        """
        recommendations = {
            "Very Low Risk": "Approve with highest credit limit",
            "Low Risk": "Approve with standard credit limit",
            "Medium Risk": "Approve with reduced credit limit",
            "High Risk": "Review application manually",
            "Very High Risk": "Decline application"
        }
        return recommendations.get(risk_level, "Review application manually")

    def run(self, host="0.0.0.0", port=8000):
        """
        Run the API server.
        
        Parameters:
        host (str): Host address
        port (int): Port number
        """
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    model = CreditScoringModel(model_type='rf')
    model.load_model('model.joblib', 'scaler.joblib')
    
    api = CreditScoringAPI(model)
    api.run()