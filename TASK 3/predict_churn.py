import pandas as pd
import numpy as np
import joblib
import sys
import os

def predict_churn(input_file, output_file=None):
    """
    Predict customer churn using the trained model.
    
    Parameters:
    -----------
    input_file : str
        Path to the CSV file containing customer data
    output_file : str, optional
        Path to save the predictions. If None, prints to console
    
    Returns:
    --------
    DataFrame with original data and churn predictions
    """
    print(f"Loading data from {input_file}...")
    
    try:
        # Load the data
        df = pd.read_csv(input_file)
        
        # Check if required columns are present
        required_columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
                           'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        
        # Remove non-predictive columns if they exist
        for col in ['RowNumber', 'CustomerId', 'Surname', 'Exited']:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Load the model
        print("Loading the trained model...")
        model_path = 'models/final_churn_model.pkl'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        model = joblib.load(model_path)
        
        # Make predictions
        print("Making predictions...")
        churn_proba = model.predict_proba(df)[:, 1]
        churn_pred = model.predict(df)
        
        # Add predictions to the dataframe
        result_df = pd.read_csv(input_file)  # Reload to keep all original columns
        result_df['ChurnProbability'] = churn_proba
        result_df['ChurnPrediction'] = churn_pred
        
        # Save or print results
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        else:
            print("\nPrediction Results:")
            print(result_df[['ChurnProbability', 'ChurnPrediction']].head())
            print(f"\nTotal customers: {len(result_df)}")
            print(f"Predicted to churn: {sum(churn_pred)} ({sum(churn_pred)/len(churn_pred)*100:.2f}%)")
        
        return result_df
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    """
    Main function to run the script from command line
    """
    if len(sys.argv) < 2:
        print("Usage: python predict_churn.py <input_file> [output_file]")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    predict_churn(input_file, output_file)

if __name__ == "__main__":
    main()