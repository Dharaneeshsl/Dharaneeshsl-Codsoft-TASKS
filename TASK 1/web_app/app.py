"""
Flask Web Application for Movie Genre Classification
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction import GenrePredictor
from data_preprocessing import DataPreprocessor

app = Flask(__name__)

# Initialize predictor
try:
    predictor = GenrePredictor()
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    predictor = None
    models_loaded = False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for genre prediction"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please train the models first.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        model_name = data.get('model', 'svm')
        
        # Validate input
        is_valid, message = predictor.validate_text(text)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Make prediction
        genre, confidence = predictor.predict_genre(text, model_name)
        
        # Get top predictions
        top_predictions = predictor.get_top_predictions(text, top_k=3)
        
        # Analyze features
        features = predictor.analyze_text_features(text)
        
        return jsonify({
            'success': True,
            'prediction': {
                'genre': genre,
                'confidence': round(confidence, 4),
                'model': model_name
            },
            'top_predictions': top_predictions,
            'features': features[:5]  # Top 5 features
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_all', methods=['POST'])
def predict_all():
    """API endpoint for prediction with all models"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please train the models first.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        # Validate input
        is_valid, message = predictor.validate_text(text)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # Get predictions from all models
        all_predictions = predictor.predict_with_all_models(text)
        
        return jsonify({
            'success': True,
            'predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """API endpoint to get available models"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        model_info = predictor.get_model_info()
        return jsonify({
            'success': True,
            'models': model_info
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded
    })

if __name__ == '__main__':
    print("Starting Movie Genre Classification Web App...")
    print("Make sure you have trained the models first using: python main.py --mode train")
    app.run(debug=True, host='0.0.0.0', port=5000) 