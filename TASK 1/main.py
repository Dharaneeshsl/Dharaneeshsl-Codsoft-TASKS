#!/usr/bin/env python3
"""
Movie Genre Classification - Main Entry Point
"""

import argparse
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from prediction import GenrePredictor

def main():
    parser = argparse.ArgumentParser(description='Movie Genre Classification')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate'], 
                       required=True, help='Mode to run the application')
    parser.add_argument('--text', type=str, help='Movie plot text for prediction')
    parser.add_argument('--model', choices=['naive_bayes', 'logistic_regression', 'svm'], 
                       default='svm', help='Model to use for prediction')
    parser.add_argument('--data_path', type=str, default='data/movies.csv',
                       help='Path to the dataset')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting model training...")
        
        # Initialize data preprocessor
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(args.data_path)
        
        # Train models
        trainer = ModelTrainer()
        print("Training models...")
        models = trainer.train_all_models(X_train, y_train)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        print("Evaluating models...")
        results = evaluator.evaluate_all_models(models, X_test, y_test)
        
        print("Training completed!")
        print("Model accuracies:")
        for model_name, accuracy in results.items():
            print(f"{model_name}: {accuracy:.4f}")
    
    elif args.mode == 'predict':
        if not args.text:
            print("Error: Please provide movie plot text using --text argument")
            sys.exit(1)
        
        print("Making prediction...")
        predictor = GenrePredictor()
        genre, confidence = predictor.predict_genre(args.text, args.model)
        
        print(f"Predicted Genre: {genre}")
        print(f"Confidence: {confidence:.4f}")
    
    elif args.mode == 'evaluate':
        print("Evaluating existing models...")
        
        # Load preprocessed data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(args.data_path)
        
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