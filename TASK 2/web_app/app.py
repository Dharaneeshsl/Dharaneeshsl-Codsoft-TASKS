"""
Flask Web Application for Credit Card Fraud Detection
"""

from flask import Flask, render_template, request, jsonify
import sys
import os
import json
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import FraudPredictor

app = Flask(__name__)
predictor = None

def initialize_predictor():
    """Initialize the fraud predictor"""
    global predictor
    try:
        predictor = FraudPredictor()
        return True
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for fraud prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['amt', 'lat', 'long']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        prediction, confidence = predictor.predict_fraud(data)
        
        # Get comprehensive analysis
        analysis = predictor.analyze_transaction(data)
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(confidence),
            'is_fraudulent': bool(prediction),
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch fraud prediction"""
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions data provided'}), 400
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        # Make batch predictions
        predictions, confidences = predictor.predict_batch(transactions)
        
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            results.append({
                'transaction_id': i,
                'prediction': int(pred),
                'confidence': float(conf),
                'is_fraudulent': bool(pred)
            })
        
        return jsonify({
            'results': results,
            'total_transactions': len(transactions),
            'fraudulent_count': sum(predictions),
            'legitimate_count': len(predictions) - sum(predictions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint for comprehensive transaction analysis"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get comprehensive analysis
        analysis = predictor.analyze_transaction(data)
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """API endpoint for prediction explanation"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        model_name = data.get('model', 'random_forest')
        
        # Get prediction explanation
        explanation = predictor.explain_prediction(data, model_name)
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """API endpoint to get available models"""
    try:
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized'}), 500
        
        models = list(predictor.models.keys())
        return jsonify({'models': models})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        if predictor is None:
            return jsonify({'status': 'error', 'message': 'Predictor not initialized'}), 500
        
        model_count = len(predictor.models)
        return jsonify({
            'status': 'healthy',
            'models_loaded': model_count,
            'available_models': list(predictor.models.keys())
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize predictor
    if initialize_predictor():
        print("Fraud detection system initialized successfully!")
        print("Available models:", list(predictor.models.keys()) if predictor else "None")
    else:
        print("Warning: Failed to initialize fraud detection system")
        print("Please ensure models are trained first")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000) 