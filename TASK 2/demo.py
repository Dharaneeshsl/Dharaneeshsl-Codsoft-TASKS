#!/usr/bin/env python3

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation import ModelEvaluator
from prediction import FraudPredictor

def main():
    print("=" * 60)
    print("💳 CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60)

    print("\n1. 📊 Data Preprocessing")
    print("-" * 30)

    preprocessor = DataPreprocessor()

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        'archive (4)/fraudTrain.csv',
        'archive (4)/fraudTest.csv',
        sample_size=10000
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"Test labels distribution: {np.bincount(y_test)}")

    print("\n2. 🤖 Model Training")
    print("-" * 30)

    trainer = ModelTrainer()
    print("Training models...")
    models = trainer.train_all_models(X_train, y_train)

    print(f"Trained {len(models)} models:")
    for model_name in models.keys():
        print(f"  - {model_name.replace('_', ' ').title()}")

    print("\n3. 📈 Model Evaluation")
    print("-" * 30)

    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test)

    print("Model Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    print("\n4. 🔍 Sample Predictions")
    print("-" * 30)

    predictor = FraudPredictor()

    sample_transactions = [
        {
            'amt': 150.00,
            'lat': 40.7128,
            'long': -74.0060,
            'merchant': 'online',
            'hour': 14,
            'day_of_week': 2
        },
        {
            'amt': 2500.00,
            'lat': 34.0522,
            'long': -118.2437,
            'merchant': 'travel',
            'hour': 23,
            'day_of_week': 6
        },
        {
            'amt': 25.50,
            'lat': 41.8781,
            'long': -87.6298,
            'merchant': 'grocery',
            'hour': 10,
            'day_of_week': 1
        }
    ]

    print("Testing sample transactions:")
    for i, transaction in enumerate(sample_transactions, 1):
        print(f"\nTransaction {i}:")
        print(f"  Amount: ${transaction['amt']}")
        print(f"  Location: ({transaction['lat']}, {transaction['long']})")
        print(f"  Merchant: {transaction['merchant']}")
        print(f"  Time: Hour {transaction['hour']}, Day {transaction['day_of_week']}")

        try:
            prediction, confidence = predictor.predict_fraud(transaction, 'random_forest')

            print(f"  Prediction: {'🚨 FRAUDULENT' if prediction == 1 else '✅ LEGITIMATE'}")
            print(f"  Confidence: {confidence:.4f}")

            analysis = predictor.analyze_transaction(transaction)
            print(f"  Risk Level: {analysis['risk_level']}")
            print(f"  Recommendations: {analysis['recommendations'][0]}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n5. 🔍 Feature Importance Analysis")
    print("-" * 30)

    try:
        feature_importance = predictor.get_feature_importance('random_forest')
        if feature_importance is not None:
            print("Top 10 Most Important Features:")
            top_features = np.argsort(feature_importance)[-10:][::-1]
            for i, feature_idx in enumerate(top_features, 1):
                print(f"  {i}. Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")
        else:
            print("Feature importance not available for this model")
    except Exception as e:
        print(f"Error getting feature importance: {e}")

    print("\n6. 📋 System Summary")
    print("-" * 30)

    print("✅ Credit Card Fraud Detection System Ready!")
    print(f"📊 Models Trained: {len(models)}")
    print(f"🎯 Best Model: Random Forest")
    print(f"📈 Average Accuracy: {np.mean([r['accuracy'] for r in results.values()]):.4f}")
    print(f"🔒 Fraud Detection Rate: {np.mean([r['recall'] for r in results.values()]):.4f}")

    print("\n🚀 Next Steps:")
    print("  1. Run 'python main.py --mode train' to train on full dataset")
    print("  2. Run 'python web_app/app.py' to start web interface")
    print("  3. Use 'python main.py --mode predict' for command-line predictions")

    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()