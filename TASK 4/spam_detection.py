#!/usr/bin/env python3
# Spam SMS Detection
# This script builds and evaluates multiple models for classifying SMS messages as spam or ham (legitimate)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
print(f"Dataset shape: {df.shape}")

# Display the first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# The dataset has columns v1 (label) and v2 (message), and some unnamed columns
# Let's rename the columns for clarity
df = df.rename(columns={'v1': 'label', 'v2': 'message'})

# Drop the unnamed columns as they contain NaN values
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Basic statistics
print("\nClass distribution:")
print(df['label'].value_counts())
print(f"Percentage of spam messages: {df[df['label'] == 'spam'].shape[0] / df.shape[0] * 100:.2f}%")

# Convert labels to binary (0 for ham, 1 for spam)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Analyze message length
df['message_length'] = df['message'].apply(len)
print("\nMessage length statistics:")
print(df.groupby('label')['message_length'].describe())

# Visualize message length distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
plt.title('Message Length Distribution')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.savefig('message_length_distribution.png')
plt.close()

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing stopwords
    4. Removing numbers
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text

# Apply preprocessing to the messages
print("\nPreprocessing text data...")
df['processed_message'] = df['message'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_message'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_num']  # Maintain class distribution
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create a function to build and evaluate models
def build_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Build a pipeline with TF-IDF vectorization and the specified model,
    train it, and evaluate its performance.
    """
    print(f"\n{'-'*50}")
    print(f"Building and evaluating {name}...")
    
    # Create a pipeline with TF-IDF vectorization and the model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
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
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return pipeline, accuracy

# Build and evaluate models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Support Vector Machine': LinearSVC(max_iter=1000)
}

results = {}
for name, model in models.items():
    pipeline, accuracy = build_and_evaluate_model(name, model, X_train, X_test, y_train, y_test)
    results[name] = {
        'pipeline': pipeline,
        'accuracy': accuracy
    }

# Compare model performances
print("\n" + "="*50)
print("Model Performance Comparison:")
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f}")

# Find the best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['pipeline']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

# Visualize model comparison
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

sns.barplot(x=model_names, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)  # Adjust as needed
plt.savefig('model_comparison.png')
plt.close()

# Example predictions with the best model
print("\n" + "="*50)
print("Example Predictions with the Best Model:")

example_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
    "Hey, what time are we meeting for dinner tonight?",
    "URGENT: Your bank account has been suspended. Click here to verify your information.",
    "Don't forget to pick up milk on your way home.",
    "Free entry to the biggest show in town! Limited tickets available. Reply YES to claim yours now!"
]

# Preprocess example messages
processed_examples = [preprocess_text(msg) for msg in example_messages]

# Make predictions
predictions = best_model.predict(processed_examples)

# Display results
for i, (message, prediction) in enumerate(zip(example_messages, predictions)):
    print(f"\nMessage {i+1}: {message}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")

print("\n" + "="*50)
print("Spam Detection Model Training Complete!")