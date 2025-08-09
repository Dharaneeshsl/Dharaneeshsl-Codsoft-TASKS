"""
Prediction Module for Movie Genre Classification
"""

import pickle
import os
import numpy as np
from data_preprocessing import DataPreprocessor

class GenrePredictor:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.load_components()
    
    def load_components(self):
        """
        Load the preprocessor and models
        """
        # Load preprocessor components
        if not self.preprocessor.load_preprocessor():
            raise Exception("Preprocessor components not found. Please train the model first.")
        
        # Load models
        model_names = ['naive_bayes', 'logistic_regression', 'svm', 'random_forest']
        
        for model_name in model_names:
            model_path = f'models/{model_name}_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
    
    def predict_genre(self, text, model_name='svm'):
        """
        Predict genre for a given text
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Clean the text
        clean_text = self.preprocessor.clean_text(text)
        
        # Vectorize the text
        text_vectorized = self.preprocessor.vectorizer.transform([clean_text])
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(text_vectorized)[0]
        
        # Get confidence score if available
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vectorized)[0]
            confidence = np.max(proba)
        
        # Decode the prediction
        genre = self.preprocessor.label_encoder.inverse_transform([prediction])[0]
        
        return genre, confidence
    
    def predict_with_all_models(self, text):
        """
        Predict genre using all available models
        """
        results = {}
        
        for model_name in self.models.keys():
            try:
                genre, confidence = self.predict_genre(text, model_name)
                results[model_name] = {
                    'genre': genre,
                    'confidence': confidence
                }
            except Exception as e:
                results[model_name] = {
                    'genre': 'Error',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def get_top_predictions(self, text, top_k=3):
        """
        Get top-k predictions with confidence scores
        """
        if 'svm' not in self.models:
            raise ValueError("SVM model not found. Please train the model first.")
        
        # Clean and vectorize text
        clean_text = self.preprocessor.clean_text(text)
        text_vectorized = self.preprocessor.vectorizer.transform([clean_text])
        
        # Get probability predictions
        model = self.models['svm']
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vectorized)[0]
            
            # Get top-k indices
            top_indices = np.argsort(proba)[::-1][:top_k]
            
            # Get corresponding genres and probabilities
            top_predictions = []
            for idx in top_indices:
                genre = self.preprocessor.label_encoder.inverse_transform([idx])[0]
                confidence = proba[idx]
                top_predictions.append({
                    'genre': genre,
                    'confidence': confidence
                })
            
            return top_predictions
        else:
            # Fallback to single prediction
            genre, confidence = self.predict_genre(text, 'svm')
            return [{'genre': genre, 'confidence': confidence}]
    
    def analyze_text_features(self, text):
        """
        Analyze the key features/words that influenced the prediction
        """
        # Clean the text
        clean_text = self.preprocessor.clean_text(text)
        
        # Vectorize the text
        text_vectorized = self.preprocessor.vectorizer.transform([clean_text])
        
        # Get feature names
        feature_names = self.preprocessor.vectorizer.get_feature_names_out()
        
        # Get feature importance (for SVM, we can use the coefficients)
        if 'svm' in self.models:
            model = self.models['svm']
            if hasattr(model, 'coef_'):
                coefficients = model.coef_[0]  # Assuming binary classification or first class
                
                # Get the feature values for this text
                feature_values = text_vectorized.toarray()[0]
                
                # Calculate feature importance (coefficient * feature_value)
                feature_importance = coefficients * feature_values
                
                # Get top features
                top_indices = np.argsort(np.abs(feature_importance))[::-1][:10]
                
                features = []
                for idx in top_indices:
                    if feature_values[idx] > 0:  # Only include features present in the text
                        features.append({
                            'feature': feature_names[idx],
                            'importance': feature_importance[idx],
                            'value': feature_values[idx]
                        })
                
                return features
        
        return []
    
    def get_model_info(self):
        """
        Get information about available models
        """
        info = {}
        
        for model_name, model in self.models.items():
            info[model_name] = {
                'type': type(model).__name__,
                'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A'
            }
        
        return info
    
    def validate_text(self, text):
        """
        Validate if the input text is suitable for prediction
        """
        if not text or len(text.strip()) == 0:
            return False, "Text cannot be empty"
        
        if len(text.strip()) < 10:
            return False, "Text is too short (minimum 10 characters)"
        
        if len(text.strip()) > 10000:
            return False, "Text is too long (maximum 10,000 characters)"
        
        return True, "Text is valid"
    
    def predict_batch(self, texts, model_name='svm'):
        """
        Predict genres for a batch of texts
        """
        if not texts:
            return []
        
        # Clean all texts
        clean_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Vectorize all texts
        texts_vectorized = self.preprocessor.vectorizer.transform(clean_texts)
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(texts_vectorized)
        
        # Get confidence scores if available
        confidences = []
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(texts_vectorized)
            confidences = [np.max(proba) for proba in probas]
        else:
            confidences = [0.0] * len(predictions)
        
        # Decode predictions
        genres = self.preprocessor.label_encoder.inverse_transform(predictions)
        
        # Return results
        results = []
        for i, (genre, confidence) in enumerate(zip(genres, confidences)):
            results.append({
                'text': texts[i],
                'genre': genre,
                'confidence': confidence
            })
        
        return results 