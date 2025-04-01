import numpy as np
import sys

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
        print(f"Starting validation of input data: {data}", file=sys.stderr)
        
        if not isinstance(data, dict):
            raise ValueError(f"Input data must be a dictionary, got {type(data)}")
            
        # Check for required fields
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if data[field] is None:
                raise ValueError(f"Field {field} cannot be None")
            
            # Convert to appropriate type and validate
            try:
                if field in self.integer_fields:
                    value = int(float(data[field]))  # Handle float inputs for integer fields
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
                print(f"Validated field {field}: {value}", file=sys.stderr)
            except (ValueError, TypeError) as e:
                print(f"Error validating field {field}: {str(e)}", file=sys.stderr)
                raise ValueError(f"Invalid value for {field}: {data[field]}")
        
        print("Input validation completed successfully", file=sys.stderr)
        
    def predict(self, data):
        """Predict heart failure risk"""
        try:
            print("Starting prediction with data:", data, file=sys.stderr)
            
            # Validate input
            self.validate_input(data)
            
            # Count risk factors
            risk_score = 0
            
            # Age risk
            if data['age'] > self.risk_factors['age']:
                risk_score += 1
                print("Age risk factor added", file=sys.stderr)
                
            # Blood pressure risk
            if data['trestbps'] > self.risk_factors['trestbps']:
                risk_score += 1
                print("Blood pressure risk factor added", file=sys.stderr)
                
            # Cholesterol risk
            if data['chol'] > self.risk_factors['chol']:
                risk_score += 1
                print("Cholesterol risk factor added", file=sys.stderr)
                
            # Heart rate risk
            if data['thalach'] < self.risk_factors['thalach']:
                risk_score += 1
                print("Heart rate risk factor added", file=sys.stderr)
                
            # ST depression risk
            if data['oldpeak'] > self.risk_factors['oldpeak']:
                risk_score += 1
                print("ST depression risk factor added", file=sys.stderr)
                
            # Additional risk factors
            if data['sex'] == 1:  # male
                risk_score += 1
                print("Male sex risk factor added", file=sys.stderr)
            if data['cp'] > 0:    # chest pain
                risk_score += 1
                print("Chest pain risk factor added", file=sys.stderr)
            if data['fbs'] == 1:  # high blood sugar
                risk_score += 1
                print("High blood sugar risk factor added", file=sys.stderr)
            if data['exang'] == 1: # exercise induced angina
                risk_score += 1
                print("Exercise induced angina risk factor added", file=sys.stderr)
            if data['ca'] > 0:    # number of major vessels
                risk_score += 1
                print("Major vessels risk factor added", file=sys.stderr)
                
            # Calculate probability based on risk score
            # Max score is 10 (added ca as a risk factor), convert to probability
            probability = risk_score / 10.0
            
            # Format response to match frontend expectations
            prediction = 1 if probability > 0.5 else 0
            message = "High risk of heart failure detected. Please consult a healthcare professional." if prediction == 1 else "Low risk of heart failure detected. Maintain a healthy lifestyle."
            
            result = {
                'prediction': prediction,
                'probability': probability,
                'message': message,
                'risk_score': risk_score
            }
            
            print("Prediction completed successfully:", result, file=sys.stderr)
            return result
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}", file=sys.stderr)
            raise ValueError(f"Error in prediction: {str(e)}") 