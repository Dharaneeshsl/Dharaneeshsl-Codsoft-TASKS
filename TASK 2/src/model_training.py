"""
Model Training Module for Credit Card Fraud Detection
"""

import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=1.0,
                    class_weight='balanced'
                ),
                'name': 'Logistic Regression'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced'
                ),
                'name': 'Decision Tree'
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'name': 'Random Forest'
            },
            'svm': {
                'model': SVC(
                    kernel='rbf',
                    random_state=42,
                    probability=True,
                    class_weight='balanced'
                ),
                'name': 'Support Vector Machine'
            },
            'xgboost': {
                'model': XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    scale_pos_weight=10,  # Handle class imbalance
                    n_jobs=-1
                ),
                'name': 'XGBoost'
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
            try:
                self.train_model(model_name, X_train, y_train)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
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
            try:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                performance[model_name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'model': model
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        return performance
    
    def get_best_model(self, X_test, y_test):
        """
        Get the best performing model based on accuracy
        """
        performance = self.get_model_performance(X_test, y_test)
        
        if not performance:
            return None, 0.0
        
        best_model = max(performance.items(), key=lambda x: x[1]['accuracy'])
        
        return best_model[0], best_model[1]['accuracy']
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """
        Perform cross-validation for a specific model
        """
        from sklearn.model_selection import cross_val_score
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.model_configs[model_name]['model']
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores for {model_name}:")
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores 