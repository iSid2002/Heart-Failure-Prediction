from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import json
from pathlib import Path
from model.heart_model import HeartFailureModel

app = Flask(__name__)
CORS(app)

# Initialize model
model = HeartFailureModel()
model_path = Path("model/trained_model.joblib")

# Load the trained model
if model_path.exists():
    model.load_model(str(model_path))
else:
    raise Exception("Model not found. Please train the model first.")

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction, probability = model.predict(input_data)
        
        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0]),
            'risk_level': 'High' if prediction[0] == 1 else 'Low'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Endpoint for getting feature importance data."""
    try:
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        # Convert to list of dictionaries
        importance_list = importance_df.to_dict('records')
        
        return jsonify(importance_list)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Endpoint for getting model performance metrics."""
    try:
        metrics_path = Path("model/metrics.json")
        
        if not metrics_path.exists():
            return jsonify({'error': 'Metrics not found'}), 404
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/feature-importance-plot', methods=['GET'])
def get_feature_importance_plot():
    """Endpoint for getting the feature importance plot."""
    try:
        plot_path = Path("model/feature_importance.png")
        
        if not plot_path.exists():
            return jsonify({'error': 'Plot not found'}), 404
        
        return send_from_directory(
            plot_path.parent,
            plot_path.name,
            mimetype='image/png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 