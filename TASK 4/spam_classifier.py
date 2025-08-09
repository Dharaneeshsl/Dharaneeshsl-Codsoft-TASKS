#!/usr/bin/env python3

import joblib
import re
import string
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def classify_sms(message, model_path='best_spam_detection_model.pkl'):
    model = joblib.load(model_path)
    processed_message = preprocess_text(message)
    prediction = model.predict([processed_message])[0]
    result = "Spam" if prediction == 1 else "Ham (Legitimate)"
    return result

def main():
    print("=" * 50)
    print("SMS Spam Classifier")
    print("=" * 50)
    example_messages = [
        "Congratulations! You've won a $1000 gift card. Call now to claim your prize!",
        "Hey, what time are we meeting for dinner tonight?",
        "URGENT: Your bank account has been suspended. Click here to verify your information.",
        "Don't forget to pick up milk on your way home.",
        "Free entry to the biggest show in town! Limited tickets available. Reply YES to claim yours now!"
    ]
    print("\nExample Classifications:")
    for i, message in enumerate(example_messages):
        result = classify_sms(message)
        print(f"\nMessage {i+1}: {message}")
        print(f"Classification: {result}")
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