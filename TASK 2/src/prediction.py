"""
Prediction Module for Credit Card Fraud Detection
"""

import pickle
import numpy as np
import pandas as pd
import json
import os
from data_preprocessing import DataPreprocessor

class FraudPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.load_models()
        self.load_preprocessor()
    
    def load_models(self):
        """
        Load all trained models
        """
        model_files = [
            'logistic_regression_model.pkl',
            'decision_tree_model.pkl',
            'random_forest_model.pkl',
            'svm_model.pkl',
            'xgboost_model.pkl'
        ]
        
        for model_file in model_files:
            model_path = f'models/{model_file}'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = pickle.load(f)
        
        print(f"Loaded {len(self.models)} models")
    
    def load_preprocessor(self):
        """
        Load preprocessor components
        """
        self.preprocessor.load_preprocessor()
    
    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction for prediction
        """
        # Convert to DataFrame if it's a dictionary
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, str):
            # Try to parse JSON string
            try:
                data = json.loads(transaction_data)
                df = pd.DataFrame([data])
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string")
        else:
            df = transaction_data
        
        # Apply feature engineering
        df = self.preprocessor.feature_engineering(df)
        
        # Remove non-numeric columns
        df = df.select_dtypes(include=[np.number])
        
        # Fill missing values
        df = df.fillna(df.mean())
        
        # Scale features
        df_scaled = self.preprocessor.scaler.transform(df)
        
        # Select features
        df_selected = self.preprocessor.feature_selector.transform(df_scaled)
        
        return df_selected
    
    def predict_fraud(self, transaction_data, model_name='random_forest'):
        """
        Predict fraud for a single transaction
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Preprocess transaction
        X = self.preprocess_transaction(transaction_data)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else 0.5
        
        return prediction, confidence
    
    def predict_batch(self, transactions, model_name='random_forest'):
        """
        Predict fraud for multiple transactions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(transactions, list):
            df = pd.DataFrame(transactions)
        else:
            df = transactions
        
        # Preprocess transactions
        X = self.preprocess_transaction(df)
        
        # Make predictions
        model = self.models[model_name]
        predictions = model.predict(X)
        confidences = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else [0.5] * len(predictions)
        
        return predictions, confidences
    
    def get_model_confidence(self, transaction_data, model_name='random_forest'):
        """
        Get confidence scores for all models
        """
        confidences = {}
        
        for name in self.models.keys():
            try:
                prediction, confidence = self.predict_fraud(transaction_data, name)
                confidences[name] = {
                    'prediction': prediction,
                    'confidence': confidence
                }
            except Exception as e:
                confidences[name] = {
                    'prediction': None,
                    'confidence': None,
                    'error': str(e)
                }
        
        return confidences
    
    def analyze_transaction(self, transaction_data):
        """
        Comprehensive transaction analysis
        """
        # Get predictions from all models
        model_confidences = self.get_model_confidence(transaction_data)
        
        # Calculate ensemble prediction
        predictions = [conf['prediction'] for conf in model_confidences.values() if conf['prediction'] is not None]
        confidences = [conf['confidence'] for conf in model_confidences.values() if conf['confidence'] is not None]
        
        if predictions:
            ensemble_prediction = 1 if sum(predictions) > len(predictions) / 2 else 0
            avg_confidence = np.mean(confidences) if confidences else 0.5
        else:
            ensemble_prediction = 0
            avg_confidence = 0.5
        
        # Create analysis report
        analysis = {
            'transaction_data': transaction_data,
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': avg_confidence,
            'model_predictions': model_confidences,
            'risk_level': self._get_risk_level(avg_confidence),
            'recommendations': self._get_recommendations(ensemble_prediction, avg_confidence)
        }
        
        return analysis
    
    def _get_risk_level(self, confidence):
        """
        Determine risk level based on confidence
        """
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.4:
            return "LOW"
        else:
            return "VERY LOW"
    
    def _get_recommendations(self, prediction, confidence):
        """
        Get recommendations based on prediction and confidence
        """
        recommendations = []
        
        if prediction == 1:  # Fraudulent
            if confidence >= 0.8:
                recommendations.append("BLOCK TRANSACTION - High confidence fraud detected")
                recommendations.append("Flag card for review")
                recommendations.append("Notify customer immediately")
            elif confidence >= 0.6:
                recommendations.append("HOLD TRANSACTION - Medium confidence fraud detected")
                recommendations.append("Request additional verification")
                recommendations.append("Monitor for similar patterns")
            else:
                recommendations.append("FLAG FOR REVIEW - Low confidence fraud detected")
                recommendations.append("Request customer verification")
        else:  # Legitimate
            if confidence >= 0.8:
                recommendations.append("APPROVE TRANSACTION - High confidence legitimate")
            elif confidence >= 0.6:
                recommendations.append("APPROVE WITH MONITORING - Medium confidence legitimate")
                recommendations.append("Monitor for unusual patterns")
            else:
                recommendations.append("REVIEW MANUALLY - Low confidence prediction")
                recommendations.append("Consider additional verification")
        
        return recommendations
    
    def get_feature_importance(self, model_name='random_forest'):
        """
        Get feature importance for a specific model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_[0])
        else:
            return None
    
    def explain_prediction(self, transaction_data, model_name='random_forest'):
        """
        Explain the prediction using feature importance
        """
        # Get prediction
        prediction, confidence = self.predict_fraud(transaction_data, model_name)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(model_name)
        
        if feature_importance is None:
            return {
                'prediction': prediction,
                'confidence': confidence,
                'explanation': 'Feature importance not available for this model'
            }
        
        # Preprocess transaction to get feature names
        X = self.preprocess_transaction(transaction_data)
        
        # Get top contributing features
        feature_scores = list(zip(range(len(feature_importance)), feature_importance))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_features = feature_scores[:5]  # Top 5 features
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'top_contributing_features': [
                {
                    'feature_index': idx,
                    'importance': importance,
                    'value': X[0][idx] if X.shape[0] > 0 else 0
                }
                for idx, importance in top_features
            ]
        }
        
        return explanation 