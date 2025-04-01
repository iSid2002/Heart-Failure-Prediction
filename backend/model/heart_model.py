import numpy as np

class HeartFailureModel:
    def __init__(self):
        # Simple rule-based system instead of ML model
        self.risk_factors = {
            'age': 50,
            'trestbps': 130,  # resting blood pressure
            'chol': 200,      # cholesterol
            'thalach': 150,   # maximum heart rate
            'oldpeak': 1.0    # ST depression
        }
        
        # Required fields
        self.required_fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'thalach', 'exang', 'oldpeak'
        ]
        
    def validate_input(self, data):
        """Validate input data"""
        # Check for required fields
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            
            # Convert to appropriate type and validate
            try:
                if field in ['sex', 'cp', 'fbs', 'exang']:
                    value = int(data[field])
                    if field == 'sex' and value not in [0, 1]:
                        raise ValueError(f"Invalid value for {field}: must be 0 or 1")
                else:
                    value = float(data[field])
                    if value < 0:
                        raise ValueError(f"Invalid negative value for {field}")
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field}: {data[field]}")
        
    def predict(self, data):
        """Predict heart failure risk"""
        try:
            # Validate input
            self.validate_input(data)
            
            # Count risk factors
            risk_score = 0
            
            # Age risk
            if float(data['age']) > self.risk_factors['age']:
                risk_score += 1
                
            # Blood pressure risk
            if float(data['trestbps']) > self.risk_factors['trestbps']:
                risk_score += 1
                
            # Cholesterol risk
            if float(data['chol']) > self.risk_factors['chol']:
                risk_score += 1
                
            # Heart rate risk
            if float(data['thalach']) < self.risk_factors['thalach']:
                risk_score += 1
                
            # ST depression risk
            if float(data['oldpeak']) > self.risk_factors['oldpeak']:
                risk_score += 1
                
            # Additional risk factors
            if int(data['sex']) == 1:  # male
                risk_score += 1
            if int(data['cp']) > 0:    # chest pain
                risk_score += 1
            if int(data['fbs']) == 1:  # high blood sugar
                risk_score += 1
            if int(data['exang']) == 1: # exercise induced angina
                risk_score += 1
                
            # Calculate probability based on risk score
            # Max score is 9, convert to probability
            probability = risk_score / 9.0
            
            # Format response to match frontend expectations
            prediction = 1 if probability > 0.5 else 0
            message = "High risk of heart failure detected. Please consult a healthcare professional." if prediction == 1 else "Low risk of heart failure detected. Maintain a healthy lifestyle."
            
            return {
                'prediction': prediction,
                'probability': probability,
                'message': message,
                'risk_score': risk_score
            }
            
        except Exception as e:
            raise ValueError(f"Error in prediction: {str(e)}") 