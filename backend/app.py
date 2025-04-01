from flask import Flask, request, jsonify
from flask_cors import CORS
from model.heart_model import HeartFailureModel
import os
import traceback

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize the model
model = HeartFailureModel()

@app.route('/api/healthcheck')
def healthcheck():
    return jsonify({"status": "healthy"})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Log incoming data for debugging
        print("Received data:", data)
            
        prediction = model.predict(data)
        
        # Log prediction for debugging
        print("Prediction result:", prediction)
        
        return jsonify(prediction)
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # This will show in Vercel logs
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 