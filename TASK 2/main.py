#!/usr/bin/env python3
"""
Credit Card Fraud Detection - Main Entry Point
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from prediction import FraudPredictor

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       required=True, help='Mode to run the application')
    parser.add_argument('--data', type=str, help='Transaction data for prediction')
    parser.add_argument('--model', choices=['logistic_regression', 'decision_tree', 'random_forest', 'xgboost'], 
                       default='random_forest', help='Model to use for prediction')
    parser.add_argument('--train_path', type=str, default='archive (4)/fraudTrain.csv',
                       help='Path to the training dataset')
    parser.add_argument('--test_path', type=str, default='archive (4)/fraudTest.csv',
                       help='Path to the test dataset')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting fraud detection model training...")
        
        # Initialize data preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(args.train_path, args.test_path)
        
        # Train models
        trainer = ModelTrainer()
        print("Training models...")
        models = trainer.train_all_models(X_train, y_train)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        print("Evaluating models...")
        results = evaluator.evaluate_all_models(models, X_test, y_test)
        
        print("Training completed!")
        print("Model performance:")
        for model_name, metrics in results.items():
            print(f"{model_name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    elif args.mode == 'predict':
        if not args.data:
            print("Error: Please provide transaction data using --data argument")
            sys.exit(1)
        
        print("Making fraud prediction...")
        predictor = FraudPredictor()
        prediction, confidence = predictor.predict_fraud(args.data, args.model)
        
        print(f"Prediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'}")
        print(f"Confidence: {confidence:.4f}")
    
    elif args.mode == 'evaluate':
        print("Evaluating existing models...")
        
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(args.train_path, args.test_path)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(None, X_test, y_test)  # Will load saved models
        
        print("Model evaluation results:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main() 