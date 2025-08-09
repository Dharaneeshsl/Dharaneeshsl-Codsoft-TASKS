#!/usr/bin/env python3
"""
Test script to verify dataset loading and preprocessing
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor

def test_dataset_loading():
    print("🧪 Testing Dataset Loading")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test real dataset loading
    print("\n1. Testing real dataset loading...")
    try:
        train_df, test_df = preprocessor.load_real_dataset()
        
        if train_df is not None:
            print(f"✅ Successfully loaded {len(train_df)} training samples")
            print(f"✅ Successfully loaded {len(test_df)} test samples")
            print(f"✅ Number of unique genres: {train_df['genre'].nunique()}")
            
            # Show some sample data
            print("\nSample training data:")
            print(train_df.head(3))
            
            print("\nGenre distribution (top 10):")
            print(train_df['genre'].value_counts().head(10))
            
        else:
            print("❌ Failed to load real dataset")
            
    except Exception as e:
        print(f"❌ Error loading real dataset: {e}")
    
    # Test data preprocessing
    print("\n2. Testing data preprocessing...")
    try:
        X_train, X_test, y_train, y_test = preprocessor.prepare_data()
        
        print(f"✅ Training set shape: {X_train.shape}")
        print(f"✅ Test set shape: {X_test.shape}")
        print(f"✅ Number of features: {X_train.shape[1]}")
        print(f"✅ Number of classes: {len(set(y_train))}")
        
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Dataset testing completed!")

if __name__ == "__main__":
    test_dataset_loading() 