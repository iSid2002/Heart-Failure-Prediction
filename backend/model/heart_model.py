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
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal'
        ]
        
        # Define which fields should be integers
        self.integer_fields = ['sex', 'cp', 'fbs', 'exang', 'restecg', 'slope', 'ca', 'thal']
        
    def validate_input(self, data):
        """Validate input data"""
        # Check for required fields
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            
            # Convert to appropriate type and validate
            try:
                if field in self.integer_fields:
                    value = int(data[field])
                    # Validate ranges for integer fields
                    if field == 'sex' and value not in [0, 1]:
                        raise ValueError(f"Invalid value for {field}: must be 0 or 1")
                    elif field == 'cp' and not (0 <= value <= 3):
                        raise ValueError(f"Invalid value for {field}: must be between 0 and 3")
                    elif field in ['fbs', 'exang'] and value not in [0, 1]:
                        raise ValueError(f"Invalid value for {field}: must be 0 or 1")
                    elif field == 'restecg' and not (0 <= value <= 2):
                        raise ValueError(f"Invalid value for {field}: must be between 0 and 2")
                    elif field == 'slope' and not (0 <= value <= 2):
                        raise ValueError(f"Invalid value for {field}: must be between 0 and 2")
                    elif field == 'ca' and not (0 <= value <= 4):
                        raise ValueError(f"Invalid value for {field}: must be between 0 and 4")
                    elif field == 'thal' and not (0 <= value <= 3):
                        raise ValueError(f"Invalid value for {field}: must be between 0 and 3")
                else:
                    value = float(data[field])
                    if value < 0:
                        raise ValueError(f"Invalid negative value for {field}")
                # Store the converted value back
                data[field] = value
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
            if data['age'] > self.risk_factors['age']:
                risk_score += 1
                
            # Blood pressure risk
            if data['trestbps'] > self.risk_factors['trestbps']:
                risk_score += 1
                
            # Cholesterol risk
            if data['chol'] > self.risk_factors['chol']:
                risk_score += 1
                
            # Heart rate risk
            if data['thalach'] < self.risk_factors['thalach']:
                risk_score += 1
                
            # ST depression risk
            if data['oldpeak'] > self.risk_factors['oldpeak']:
                risk_score += 1
                
            # Additional risk factors
            if data['sex'] == 1:  # male
                risk_score += 1
            if data['cp'] > 0:    # chest pain
                risk_score += 1
            if data['fbs'] == 1:  # high blood sugar
                risk_score += 1
            if data['exang'] == 1: # exercise induced angina
                risk_score += 1
            if data['ca'] > 0:    # number of major vessels
                risk_score += 1
                
            # Calculate probability based on risk score
            # Max score is 10 (added ca as a risk factor), convert to probability
            probability = risk_score / 10.0
            
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