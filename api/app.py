from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from model.heart_model import HeartFailureModel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model = HeartFailureModel()
model.load_model(project_root / 'model' / 'heart_failure_model.joblib')

# Store predictions in memory (in a real app, this would be in a database)
predictions = []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Create DataFrame from input
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction, probability = model.predict(input_data)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'message': 'High risk of heart disease' if prediction[0] == 1 else 'Low risk of heart disease',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save prediction to history
        prediction_record = {
            'input_data': data,
            'result': response
        }
        predictions.append(prediction_record)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Return the prediction history."""
    return jsonify(predictions)

@app.route('/predictions/<int:index>', methods=['DELETE'])
def delete_prediction(index):
    """Delete a prediction from history."""
    if 0 <= index < len(predictions):
        predictions.pop(index)
        return jsonify({'message': 'Prediction deleted successfully'})
    return jsonify({'error': 'Invalid prediction index'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/feature_ranges', methods=['GET'])
def get_feature_ranges():
    """Return the expected ranges and descriptions for each feature."""
    feature_info = {
        'age': {
            'type': 'numeric',
            'description': 'Age in years',
            'range': [0, 150]
        },
        'sex': {
            'type': 'categorical',
            'description': 'Gender (1 = male; 0 = female)',
            'values': [0, 1]
        },
        'cp': {
            'type': 'categorical',
            'description': 'Chest pain type',
            'values': [0, 1, 2, 3],
            'value_descriptions': {
                0: 'Typical angina',
                1: 'Atypical angina',
                2: 'Non-anginal pain',
                3: 'Asymptomatic'
            }
        },
        'trestbps': {
            'type': 'numeric',
            'description': 'Resting blood pressure (in mm Hg)',
            'range': [94, 200]
        },
        'chol': {
            'type': 'numeric',
            'description': 'Serum cholesterol in mg/dl',
            'range': [126, 564]
        },
        'fbs': {
            'type': 'categorical',
            'description': 'Fasting blood sugar > 120 mg/dl',
            'values': [0, 1],
            'value_descriptions': {
                0: 'False',
                1: 'True'
            }
        },
        'restecg': {
            'type': 'categorical',
            'description': 'Resting electrocardiographic results',
            'values': [0, 1, 2],
            'value_descriptions': {
                0: 'Normal',
                1: 'Having ST-T wave abnormality',
                2: 'Showing probable or definite left ventricular hypertrophy'
            }
        },
        'thalach': {
            'type': 'numeric',
            'description': 'Maximum heart rate achieved',
            'range': [71, 202]
        },
        'exang': {
            'type': 'categorical',
            'description': 'Exercise induced angina',
            'values': [0, 1],
            'value_descriptions': {
                0: 'No',
                1: 'Yes'
            }
        },
        'oldpeak': {
            'type': 'numeric',
            'description': 'ST depression induced by exercise relative to rest',
            'range': [0, 6.2]
        },
        'slope': {
            'type': 'categorical',
            'description': 'Slope of the peak exercise ST segment',
            'values': [0, 1, 2],
            'value_descriptions': {
                0: 'Upsloping',
                1: 'Flat',
                2: 'Downsloping'
            }
        },
        'ca': {
            'type': 'categorical',
            'description': 'Number of major vessels colored by flourosopy',
            'values': [0, 1, 2, 3]
        },
        'thal': {
            'type': 'categorical',
            'description': 'Thalassemia',
            'values': [0, 1, 2, 3],
            'value_descriptions': {
                0: 'Normal',
                1: 'Fixed defect',
                2: 'Reversable defect',
                3: 'Unknown'
            }
        }
    }
    
    return jsonify(feature_info)

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Return the feature importance data."""
    feature_importance = [
        {'feature': 'Number of Major Vessels', 'importance': 0.1469},
        {'feature': 'Chest Pain Type', 'importance': 0.1457},
        {'feature': 'Thalassemia', 'importance': 0.1338},
        {'feature': 'ST Depression', 'importance': 0.1106},
        {'feature': 'Maximum Heart Rate', 'importance': 0.0893},
        {'feature': 'Age', 'importance': 0.0779},
        {'feature': 'Exercise-Induced Angina', 'importance': 0.0725},
        {'feature': 'ST Slope', 'importance': 0.0586},
        {'feature': 'Cholesterol', 'importance': 0.0568},
        {'feature': 'Blood Pressure', 'importance': 0.0503},
        {'feature': 'Gender', 'importance': 0.0406},
        {'feature': 'ECG Results', 'importance': 0.0121},
        {'feature': 'Fasting Blood Sugar', 'importance': 0.0049}
    ]
    return jsonify(feature_importance)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 