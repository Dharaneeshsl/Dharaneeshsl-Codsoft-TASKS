"""
Model Training Module for Movie Genre Classification
"""

import pickle
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'naive_bayes': {
                'model': MultinomialNB(alpha=1.0),
                'name': 'Naive Bayes'
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    C=1.0,
                    solver='liblinear'
                ),
                'name': 'Logistic Regression'
            },
            'svm': {
                'model': SVC(
                    kernel='linear',
                    random_state=42,
                    probability=True,
                    C=1.0
                ),
                'name': 'Support Vector Machine'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10
                ),
                'name': 'Random Forest'
            }
        }
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Training {self.model_configs[model_name]['name']}...")
        
        model = self.model_configs[model_name]['model']
        model.fit(X_train, y_train)
        
        self.models[model_name] = model
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all available models
        """
        print("Training all models...")
        
        for model_name in self.model_configs.keys():
            self.train_model(model_name, X_train, y_train)
        
        # Save all trained models
        self.save_models()
        
        return self.models
    
    def save_models(self):
        """
        Save all trained models to disk
        """
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = f'models/{model_name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            print(f"Saved {model_name} model to {model_path}")
    
    def load_models(self):
        """
        Load all saved models from disk
        """
        loaded_models = {}
        
        for model_name in self.model_configs.keys():
            model_path = f'models/{model_name}_model.pkl'
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    loaded_models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model from {model_path}")
            else:
                print(f"Model file not found: {model_path}")
        
        return loaded_models
    
    def get_model_performance(self, X_test, y_test):
        """
        Get performance metrics for all models
        """
        if not self.models:
            self.models = self.load_models()
        
        performance = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            performance[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
        
        return performance
    
    def get_best_model(self, X_test, y_test):
        """
        Get the best performing model based on accuracy
        """
        performance = self.get_model_performance(X_test, y_test)
        
        best_model = max(performance.items(), key=lambda x: x[1]['accuracy'])
        
        return best_model[0], best_model[1]['accuracy'] 