#!/usr/bin/env python3
"""
Movie Genre Classification - Demo Script
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from prediction import GenrePredictor

def main():
    print("🎬 Movie Genre Classification Demo")
    print("=" * 50)
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Step 2: Train models
    print("\n2. Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train)
    print(f"   Models trained: {len(models)}")
    
    # Step 3: Evaluate models
    print("\n3. Evaluating models...")
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test)
    
    print("\n   Model Performance:")
    print("   " + "-" * 50)
    for model_name, metrics in results.items():
        print(f"   {model_name:<20} Accuracy: {metrics['accuracy']:.3f}")
    
    # Step 4: Demo predictions
    print("\n4. Demo Predictions")
    print("   " + "-" * 50)
    
    predictor = GenrePredictor()
    
    sample_texts = [
        "A young wizard discovers his magical heritage and must defeat an evil sorcerer to save the world.",
        "Two detectives investigate a series of mysterious murders in a small town.",
        "A romantic comedy about two people who fall in love despite their differences.",
        "A group of friends go on a hilarious adventure that leads to unexpected consequences.",
        "A terrifying story about supernatural forces that haunt a family in their new home."
    ]
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n   Example {i}:")
        print(f"   Text: {text}")
        
        # Get prediction from best model
        genre, confidence = predictor.predict_genre(text, 'svm')
        print(f"   Predicted Genre: {genre}")
        print(f"   Confidence: {confidence:.3f}")
        
        # Get top 3 predictions
        top_predictions = predictor.get_top_predictions(text, top_k=3)
        print(f"   Top 3 predictions:")
        for j, pred in enumerate(top_predictions, 1):
            print(f"     {j}. {pred['genre']} ({pred['confidence']:.3f})")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nTo run the web application:")
    print("   python web_app/app.py")
    print("\nTo make predictions from command line:")
    print("   python main.py --mode predict --text 'Your movie plot here'")

if __name__ == "__main__":
    main() 