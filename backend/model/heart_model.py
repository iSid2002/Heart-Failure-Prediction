import joblib
import numpy as np
import os
import pandas as pd

class HeartFailureModel:
    def __init__(self):
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct paths to the model and scaler files
        model_path = os.path.join(current_dir, 'heart_failure_model.joblib')
        scaler_path = os.path.join(current_dir, 'scaler.joblib')
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully")
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            raise

    def predict(self, data):
        try:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([data])
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0][1]
            
            # Determine message based on prediction
            message = "High risk of heart disease" if prediction == 1 else "Low risk of heart disease"
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "message": message
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise 