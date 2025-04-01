from flask import Flask, request, jsonify
from flask_cors import CORS
from model.heart_model import HeartFailureModel
import os
import traceback
import sys
import json

app = Flask(__name__)

# Configure CORS
CORS(app)

# Initialize the model
model = HeartFailureModel()

@app.route('/api/healthcheck')
def healthcheck():
    return jsonify({"status": "healthy"})

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Log request details
        print("Request headers:", dict(request.headers), file=sys.stderr)
        print("Request data:", request.get_data(as_text=True), file=sys.stderr)
        
        if not request.is_json:
            return jsonify({
                'error': 'Request must be JSON',
                'content_type': request.content_type
            }), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Log incoming data for debugging
        print("Received data:", json.dumps(data, indent=2), file=sys.stderr)
            
        prediction = model.predict(data)
        
        # Log prediction for debugging
        print("Prediction result:", json.dumps(prediction, indent=2), file=sys.stderr)
        
        return jsonify(prediction)
    except ValueError as ve:
        error_msg = str(ve)
        print(f"Validation error: {error_msg}", file=sys.stderr)
        return jsonify({
            'error': 'Validation error',
            'message': error_msg
        }), 400
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 