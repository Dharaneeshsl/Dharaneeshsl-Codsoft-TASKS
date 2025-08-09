import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Create directory for feature importance plots
if not os.path.exists('results'):
    os.makedirs('results')

# Load the data
print("Loading data...")
df = pd.read_csv('Churn_Modelling.csv')

# Remove non-predictive columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Split features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Define preprocessing for categorical and numerical features
categorical_features = ['Geography', 'Gender']
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

# Load the trained model
model = joblib.load('models/final_churn_model.pkl')

# Get feature names after preprocessing
feature_names = []
for name, transformer, features in model.named_steps['preprocessor'].transformers_:
    if name == 'num':
        feature_names.extend(features)
    elif name == 'cat':
        for feature in features:
            try:
                categories = transformer.named_steps['onehot'].get_feature_names_out([feature])
                feature_names.extend(categories)
            except:
                # If there's an issue with getting feature names, use generic names
                feature_names.extend([f"{feature}_{i}" for i in range(len(transformer.named_steps['onehot'].categories_[0]))])

# Get feature importances
importances = model.named_steps['model'].feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(12, 8))
plt.title('Feature Importances - Gradient Boosting')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('results/feature_importance_gradient_boosting.png')

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices]
})

# Save feature importances to CSV
feature_importance_df.to_csv('results/feature_importance.csv', index=False)

print("Feature importance plot and CSV created.")