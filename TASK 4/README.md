# Spam SMS Detection System

This project implements a machine learning system for classifying SMS messages as spam or legitimate (ham). It uses various text processing techniques and machine learning algorithms to achieve high accuracy in spam detection.

## Project Overview

The system uses natural language processing techniques and machine learning algorithms to automatically identify unwanted or fraudulent SMS messages. It implements multiple classification models and compares their performance to find the most effective approach.

## Dataset

The dataset consists of 5,572 SMS messages labeled as either "spam" or "ham" (legitimate):
- Total messages: 5,572
- Ham messages: 4,825 (86.59%)
- Spam messages: 747 (13.41%)

## Features

- Text preprocessing with stopword removal and punctuation handling
- Feature extraction using TF-IDF and Count Vectorization
- Implementation of multiple machine learning models:
  - Support Vector Machines
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Naive Bayes
- Performance evaluation with accuracy, precision, recall, and F1-score
- Visualization of results with confusion matrices and ROC curves
- Feature importance analysis
- Interactive SMS classification tool

## Files in this Project

- `spam_detection.py`: Basic implementation of spam detection models
- `advanced_spam_detection.py`: Enhanced implementation with additional models and features
- `spam_classifier.py`: Interactive tool for classifying new SMS messages
- `spam_detection_report.md`: Comprehensive report on the analysis and findings
- `README.md`: This file, providing an overview of the project
- Various PNG files: Visualizations of model performance and data analysis

## Model Performance

The Support Vector Machine (SVM) with TF-IDF vectorization achieved the highest accuracy of 98.92%, with excellent precision and recall for both spam and ham messages.

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|-------|----------|------------------|---------------|-----------------|
| SVM with TF-IDF | 98.92% | 1.00 | 0.92 | 0.96 |
| Naive Bayes with Count Vectors | 97.76% | 0.97 | 0.86 | 0.91 |
| Random Forest with TF-IDF | 97.58% | 1.00 | 0.82 | 0.90 |
| Logistic Regression with TF-IDF | 96.68% | 0.98 | 0.77 | 0.86 |
| Gradient Boosting with TF-IDF | 96.23% | 0.96 | 0.75 | 0.84 |

## Key Findings

1. Spam messages tend to be significantly longer than legitimate messages
2. TF-IDF vectorization effectively captures the distinguishing features of spam messages
3. Support Vector Machines outperform other models for this task
4. The model achieves high precision, ensuring minimal false positives (legitimate messages classified as spam)
5. With 92% recall for spam detection, the SVM model successfully identifies the vast majority of unwanted messages

## Important Features for Spam Detection

The Random Forest model identified these as the top features for distinguishing spam messages:
- call
- txt
- free
- claim
- mobile
- stop
- reply
- 500
- urgent
- text
- prize
- win

## How to Use

### Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
nltk
joblib
```

### Running the Basic Model

```bash
python spam_detection.py
```

### Running the Advanced Model

```bash
python advanced_spam_detection.py
```

### Using the Interactive Classifier

```bash
python spam_classifier.py
```

## Example Usage in Python Code

```python
from spam_classifier import classify_sms

# Classify a new message
result = classify_sms("Congratulations! You've won a $1000 gift card. Call now!")
print(result)  # Output: "Spam"

result = classify_sms("Hey, what time are we meeting for dinner tonight?")
print(result)  # Output: "Ham (Legitimate)"
```

## Future Improvements

Potential enhancements to the current system:

1. Advanced feature engineering (sentiment analysis, URL detection)
2. Word embeddings (Word2Vec, GloVe)
3. Deep learning models (LSTM, transformers)
4. Ensemble methods combining multiple models
5. Regular retraining with new spam examples