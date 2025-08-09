#!/usr/bin/env python3
# Advanced Spam SMS Detection
# This script implements additional models and features for spam detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import re
import string
import nltk
from nltk.corpus import stopwords

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# Rename columns for clarity
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Drop the unnamed columns
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Convert labels to binary (0 for ham, 1 for spam)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Enhanced text preprocessing function
def advanced_preprocess(text):
    """
    Advanced preprocessing with more thorough cleaning
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Rejoin
    text = ' '.join(words)
    
    return text

# Apply advanced preprocessing
print("\nApplying advanced preprocessing...")
df['processed_message'] = df['message'].apply(advanced_preprocess)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_message'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_num']
)

# Function to evaluate model and plot ROC curve
def evaluate_model_with_roc(name, model, X_train, X_test, y_train, y_test, vectorizer):
    """
    Build a pipeline with vectorization and the specified model,
    train it, evaluate performance, and plot ROC curve.
    """
    print(f"\n{'-'*50}")
    print(f"Building and evaluating {name}...")
    
    # Create a pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # For ROC curve, we need probability estimates
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_{name.lower().replace(" ", "_")}.png')
        plt.close()
    except:
        print(f"Note: {name} doesn't support probability estimates for ROC curve.")
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'advanced_cm_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return pipeline, accuracy

# Create vectorizers
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
count_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))

# Define models
models = {
    'SVM with TF-IDF': (LinearSVC(max_iter=1000), tfidf_vectorizer),
    'Logistic Regression with TF-IDF': (LogisticRegression(max_iter=1000), tfidf_vectorizer),
    'Random Forest with TF-IDF': (RandomForestClassifier(n_estimators=100), tfidf_vectorizer),
    'Gradient Boosting with TF-IDF': (GradientBoostingClassifier(), tfidf_vectorizer),
    'Naive Bayes with Count Vectors': (MultinomialNB(), count_vectorizer)
}

# Evaluate all models
results = {}
for name, (model, vectorizer) in models.items():
    pipeline, accuracy = evaluate_model_with_roc(name, model, X_train, X_test, y_train, y_test, vectorizer)
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy
    }

# Compare model performances
print("\n" + "="*50)
print("Model Performance Comparison:")
for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name}: {result['accuracy']:.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['pipeline']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Visualize model comparison
plt.figure(figsize=(12, 8))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

# Sort by accuracy
sorted_indices = np.argsort(accuracies)[::-1]
model_names = [model_names[i] for i in sorted_indices]
accuracies = [accuracies[i] for i in sorted_indices]

sns.barplot(x=accuracies, y=model_names)
plt.title('Model Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0.9, 1.0)  # Adjust as needed
plt.tight_layout()
plt.savefig('advanced_model_comparison.png')
plt.close()

# Feature importance analysis (if applicable)
if 'Random Forest with TF-IDF' in results:
    rf_pipeline = results['Random Forest with TF-IDF']['pipeline']
    rf_model = rf_pipeline.named_steps['classifier']
    vectorizer = rf_pipeline.named_steps['vectorizer']
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Get feature importances
    importances = rf_model.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[::-1]
    
    # Print top 20 features
    print("\nTop 20 important features for Random Forest:")
    for i in range(min(20, len(feature_names))):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances (Random Forest)')
    plt.bar(range(min(30, len(feature_names))), importances[indices[:30]], align='center')
    plt.xticks(range(min(30, len(feature_names))), feature_names[indices[:30]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()

# Save the best model
import joblib
joblib.dump(best_model, 'best_spam_detection_model.pkl')
print(f"\nBest model saved as 'best_spam_detection_model.pkl'")

# Create a simple function to classify new messages
def classify_message(message, model):
    """
    Classify a new message as spam or ham using the trained model
    """
    # Preprocess the message
    processed_message = advanced_preprocess(message)
    
    # Make prediction
    prediction = model.predict([processed_message])[0]
    
    # Return result
    return "Spam" if prediction == 1 else "Ham"

# Example usage
print("\n" + "="*50)
print("Example Usage of the Spam Detection Model:")
print("\nTo classify a new message:")
print("from advanced_spam_detection import classify_message, joblib")
print("model = joblib.load('best_spam_detection_model.pkl')")
print("result = classify_message('Your message here', model)")
print("print(result)")

print("\n" + "="*50)
print("Advanced Spam Detection Model Training Complete!")