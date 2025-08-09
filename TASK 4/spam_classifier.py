#!/usr/bin/env python3
# Spam SMS Classifier
# This script demonstrates how to use the trained model to classify new SMS messages

import joblib
import re
import string
from nltk.corpus import stopwords

def preprocess_text(text):
    """
    Preprocess text by:
    1. Converting to lowercase
    2. Removing URLs
    3. Removing HTML tags
    4. Removing punctuation
    5. Removing stopwords
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

def classify_sms(message, model_path='best_spam_detection_model.pkl'):
    """
    Classify an SMS message as spam or ham using the trained model
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Preprocess the message
    processed_message = preprocess_text(message)
    
    # Make prediction
    prediction = model.predict([processed_message])[0]
    
    # Return result with confidence
    result = "Spam" if prediction == 1 else "Ham (Legitimate)"
    
    return result

def main():
    """
    Main function to demonstrate the SMS classifier
    """
    print("=" * 50)
    print("SMS Spam Classifier")
    print("=" * 50)
    
    # Example messages
    example_messages = [
        "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
        "Hey, what time are we meeting for dinner tonight?",
        "URGENT: Your bank account has been suspended. Click here to verify your information.",
        "Don't forget to pick up milk on your way home.",
        "Free entry to the biggest show in town! Limited tickets available. Reply YES to claim yours now!"
    ]
    
    # Classify each example message
    print("\nExample Classifications:")
    for i, message in enumerate(example_messages):
        result = classify_sms(message)
        print(f"\nMessage {i+1}: {message}")
        print(f"Classification: {result}")
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode")
    print("Enter an SMS message to classify (or 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter message: ")
        if user_input.lower() == 'quit':
            break
        
        result = classify_sms(user_input)
        print(f"Classification: {result}")
    
    print("\nThank you for using the SMS Spam Classifier!")

if __name__ == "__main__":
    main()