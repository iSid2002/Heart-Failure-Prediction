from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model.heart_model import HeartFailureModel
import os

app = Flask(__name__, static_folder='../frontend')

# Configure CORS
CORS(app)

# Initialize the model
model = HeartFailureModel()

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Make prediction
        result = model.predict(data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": {
                "code": "500",
                "message": str(e)
            }
        }), 500

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port) 