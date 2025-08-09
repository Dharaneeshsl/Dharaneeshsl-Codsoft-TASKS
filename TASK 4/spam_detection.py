#!/usr/bin/env python3

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

np.random.seed(42)

print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')
print(f"Dataset shape: {df.shape}")

print("\nFirst few rows of the dataset:")
print(df.head())

df = df.rename(columns={'v1': 'label', 'v2': 'message'})

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['label'].value_counts())
print(f"Percentage of spam messages: {df[df['label'] == 'spam'].shape[0] / df.shape[0] * 100:.2f}%")

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

df['message_length'] = df['message'].apply(len)
print("\nMessage length statistics:")
print(df.groupby('label')['message_length'].describe())

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True)
plt.title('Message Length Distribution')
plt.xlabel('Message Length')
plt.ylabel('Frequency')
plt.savefig('message_length_distribution.png')
plt.close()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = re.sub(r'\d+', '', text)
    return text

print("\nPreprocessing text data...")
df['processed_message'] = df['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_message'], 
    df['label_num'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label_num']
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

def build_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'-'*50}")
    print(f"Building and evaluating {name}...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    return pipeline, accuracy

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

print("\n" + "="*50)
print("Model Performance Comparison:")
for name, result in results.items():
    print(f"{name}: {result['accuracy']:.4f}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['pipeline']
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

sns.barplot(x=model_names, y=accuracies)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.savefig('model_comparison.png')
plt.close()

print("\n" + "="*50)
print("Example Predictions with the Best Model:")

example_messages = [
    "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
    "Hey, what time are we meeting for dinner tonight?",
    "URGENT: Your bank account has been suspended. Click here to verify your information.",
    "Don't forget to pick up milk on your way home.",
    "Free entry to the biggest show in town! Limited tickets available. Reply YES to claim yours now!"
]

processed_examples = [preprocess_text(msg) for msg in example_messages]

predictions = best_model.predict(processed_examples)

for i, (message, prediction) in enumerate(zip(example_messages, predictions)):
    print(f"\nMessage {i+1}: {message}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")

print("\n" + "="*50)
print("Spam Detection Model Training Complete!")