"""
Data Preprocessing Module for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)
        self.label_encoders = {}
        
    def load_data(self, train_path, test_path):
        """
        Load the credit card fraud dataset
        """
        print(f"Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path)
        
        print(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def explore_data(self, df):
        """
        Explore the dataset and show basic statistics
        """
        print("\nDataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print("\nData Types:")
        print(df.dtypes.value_counts())
        
        print("\nMissing Values:")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found!")
        
        print("\nTarget Distribution:")
        if 'is_fraud' in df.columns:
            fraud_dist = df['is_fraud'].value_counts()
            print(f"Legitimate: {fraud_dist[0]} ({fraud_dist[0]/len(df)*100:.2f}%)")
            print(f"Fraudulent: {fraud_dist[1]} ({fraud_dist[1]/len(df)*100:.2f}%)")
        
        return df
    
    def feature_engineering(self, df):
        """
        Create new features for better fraud detection
        """
        print("Performing feature engineering...")
        
        # Create time-based features
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_date_trans_time'].dt.hour
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['month'] = df['trans_date_trans_time'].dt.month
            df['is_night'] = (df['hour'] >= 22) | (df['hour'] <= 6)
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Create amount-based features
        if 'amt' in df.columns:
            df['amt_log'] = np.log1p(df['amt'])
            df['amt_sqrt'] = np.sqrt(df['amt'])
            df['high_amount'] = df['amt'] > df['amt'].quantile(0.95)
            df['low_amount'] = df['amt'] < df['amt'].quantile(0.05)
        
        # Create location-based features
        if 'lat' in df.columns and 'long' in df.columns:
            df['distance_from_home'] = np.sqrt(
                (df['lat'] - df['lat'].mean())**2 + 
                (df['long'] - df['long'].mean())**2
            )
        
        # Create merchant-based features
        if 'merchant' in df.columns:
            df['merchant_frequency'] = df.groupby('merchant')['merchant'].transform('count')
            df['high_freq_merchant'] = df['merchant_frequency'] > df['merchant_frequency'].quantile(0.9)
        
        # Create card-based features
        if 'cc_num' in df.columns:
            df['card_frequency'] = df.groupby('cc_num')['cc_num'].transform('count')
            df['high_freq_card'] = df['card_frequency'] > df['card_frequency'].quantile(0.9)
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """
        Encode categorical features
        """
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if is_training:
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # For test data, use existing encoder
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            else:
                if is_training:
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def handle_imbalanced_data(self, X, y, method='smote'):
        """
        Handle class imbalance using various techniques
        """
        print(f"Handling class imbalance using {method}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=42, sampling_strategy=0.1)  # 10% fraud ratio
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif method == 'undersample':
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=0.1)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y
        
        print(f"Original distribution: {np.bincount(y)}")
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def select_features(self, X, y, is_training=True):
        """
        Select the most important features
        """
        if is_training:
            X_selected = self.feature_selector.fit_transform(X, y)
            print(f"Selected {X_selected.shape[1]} features out of {X.shape[1]}")
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def prepare_data(self, train_path, test_path, sample_size=None):
        """
        Complete data preprocessing pipeline
        """
        # Load data
        train_df, test_df = self.load_data(train_path, test_path)
        
        # Sample data if specified (for faster processing)
        if sample_size:
            print(f"Sampling {sample_size} records for faster processing...")
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
        
        # Explore data
        print("\nExploring training data:")
        train_df = self.explore_data(train_df)
        
        # Feature engineering
        train_df = self.feature_engineering(train_df)
        test_df = self.feature_engineering(test_df)
        
        # Separate features and target
        if 'is_fraud' in train_df.columns:
            y_train = train_df['is_fraud']
            X_train = train_df.drop(['is_fraud'], axis=1)
        else:
            y_train = None
            X_train = train_df
        
        if 'is_fraud' in test_df.columns:
            y_test = test_df['is_fraud']
            X_test = test_df.drop(['is_fraud'], axis=1)
        else:
            y_test = None
            X_test = test_df
        
        # Remove non-numeric columns and handle missing values
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        
        # Fill missing values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_test.mean())
        
        # Align columns between train and test
        common_columns = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]
        
        print(f"Final training features: {X_train.shape}")
        print(f"Final test features: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select features
        if y_train is not None:
            X_train_selected = self.select_features(X_train_scaled, y_train, is_training=True)
            X_test_selected = self.select_features(X_test_scaled, y_test, is_training=False)
        else:
            X_train_selected = X_train_scaled
            X_test_selected = X_test_scaled
        
        # Handle class imbalance for training data
        if y_train is not None:
            X_train_balanced, y_train_balanced = self.handle_imbalanced_data(
                X_train_selected, y_train, method='smote'
            )
        else:
            X_train_balanced, y_train_balanced = X_train_selected, y_train
        
        # Save preprocessor components
        self.save_preprocessor()
        
        return X_train_balanced, X_test_selected, y_train_balanced, y_test
    
    def save_preprocessor(self):
        """
        Save the preprocessor components for later use
        """
        os.makedirs('models', exist_ok=True)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open('models/feature_selector.pkl', 'wb') as f:
            pickle.dump(self.feature_selector, f)
        
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
    
    def load_preprocessor(self):
        """
        Load the saved preprocessor components
        """
        try:
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('models/feature_selector.pkl', 'rb') as f:
                self.feature_selector = pickle.load(f)
            
            with open('models/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            return True
        except FileNotFoundError:
            print("Preprocessor files not found. Please train the model first.")
            return False 