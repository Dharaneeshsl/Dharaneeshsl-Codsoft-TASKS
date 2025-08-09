#!/usr/bin/env python3
"""
Test script to verify credit card fraud dataset loading and preprocessing
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor

def test_dataset_loading():
    print("🧪 Testing Credit Card Fraud Dataset")
    print("=" * 50)
    
    # Check if dataset files exist
    train_path = 'archive (4)/fraudTrain.csv'
    test_path = 'archive (4)/fraudTest.csv'
    
    print(f"\n1. Checking dataset files...")
    if os.path.exists(train_path):
        train_size = os.path.getsize(train_path) / (1024 * 1024)  # MB
        print(f"✅ Training data found: {train_size:.1f} MB")
    else:
        print(f"❌ Training data not found: {train_path}")
        return
    
    if os.path.exists(test_path):
        test_size = os.path.getsize(test_path) / (1024 * 1024)  # MB
        print(f"✅ Test data found: {test_size:.1f} MB")
    else:
        print(f"❌ Test data not found: {test_path}")
        return
    
    # Load and explore data
    print(f"\n2. Loading and exploring data...")
    try:
        preprocessor = DataPreprocessor()
        train_df, test_df = preprocessor.load_data(train_path, test_path)
        
        print(f"✅ Training data loaded: {train_df.shape}")
        print(f"✅ Test data loaded: {test_df.shape}")
        
        # Show sample data
        print(f"\n3. Sample training data:")
        print(train_df.head())
        
        print(f"\n4. Data types:")
        print(train_df.dtypes.value_counts())
        
        print(f"\n5. Missing values:")
        missing_values = train_df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("✅ No missing values found!")
        
        # Check target variable
        if 'is_fraud' in train_df.columns:
            fraud_dist = train_df['is_fraud'].value_counts()
            print(f"\n6. Target distribution:")
            print(f"   Legitimate: {fraud_dist[0]} ({fraud_dist[0]/len(train_df)*100:.2f}%)")
            print(f"   Fraudulent: {fraud_dist[1]} ({fraud_dist[1]/len(train_df)*100:.2f}%)")
            print(f"   Class imbalance ratio: {fraud_dist[0]/fraud_dist[1]:.1f}:1")
        else:
            print(f"\n6. Target variable 'is_fraud' not found in training data")
        
        # Test preprocessing pipeline
        print(f"\n7. Testing preprocessing pipeline...")
        try:
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(
                train_path, test_path, sample_size=1000
            )
            
            print(f"✅ Preprocessing successful!")
            print(f"   Training features: {X_train.shape}")
            print(f"   Test features: {X_test.shape}")
            print(f"   Training labels: {y_train.shape}")
            print(f"   Test labels: {y_test.shape}")
            
            if y_train is not None:
                print(f"   Training label distribution: {np.bincount(y_train)}")
            if y_test is not None:
                print(f"   Test label distribution: {np.bincount(y_test)}")
                
        except Exception as e:
            print(f"❌ Error in preprocessing: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 Dataset testing completed!")

if __name__ == "__main__":
    test_dataset_loading() 