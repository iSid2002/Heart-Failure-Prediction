from flask import Flask, request, jsonify
from flask_cors import CORS
from model.heart_model import HeartFailureModel
import traceback
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["https://heart-failure-prediction-frontend.onrender.com", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize the model
try:
    model = HeartFailureModel()
    print("Model initialized successfully")
except Exception as e:
    print(f"Error initializing model: {e}")
    traceback.print_exc()
    raise

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "healthy", "message": "Heart Failure Prediction API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate request data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        print("Received prediction request with data:", data)
        
        # Make prediction
        result = model.predict(data)
        print("Prediction result:", result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port) 