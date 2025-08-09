# Credit Card Fraud Detection

## Project Overview
This project implements a comprehensive machine learning system to detect fraudulent credit card transactions. The system uses multiple algorithms including Logistic Regression, Decision Trees, Random Forests, and advanced techniques to classify transactions as fraudulent or legitimate with high accuracy.

## Features
- **Data Preprocessing**: Advanced feature engineering and data cleaning
- **Multiple ML Models**: Logistic Regression, Decision Trees, Random Forest, SVM, XGBoost
- **Model Evaluation**: Comprehensive metrics and visualization
- **Real-time Detection**: Web interface for transaction analysis
- **Anomaly Detection**: Advanced techniques for fraud pattern recognition
- **Performance Optimization**: Handling imbalanced datasets

## Project Structure
```
TASK 2/
├── archive (4)/                    # Original dataset
│   ├── fraudTrain.csv             # Training data (335MB)
│   └── fraudTest.csv              # Test data (143MB)
├── data/                          # Processed data (auto-created)
├── models/                        # Trained models (auto-created)
├── notebooks/                     # Jupyter notebooks
├── src/                          # Core modules
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
├── web_app/                      # Flask web interface
│   ├── app.py
│   ├── templates/
│   └── static/
├── requirements.txt
├── main.py
├── demo.py
├── test_dataset.py
├── QUICKSTART.md
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TASK 2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python main.py --mode train
```

### Making Predictions
```bash
python main.py --mode predict --data "transaction_data.json"
```

### Running the Web App
```bash
python web_app/app.py
```

## Model Performance
- **Random Forest**: Accuracy ~99.9%, Precision ~95%
- **XGBoost**: Accuracy ~99.9%, Precision ~96%
- **Logistic Regression**: Accuracy ~99.8%, Precision ~92%
- **Decision Tree**: Accuracy ~99.7%, Precision ~90%

## Technologies Used
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Flask
- XGBoost
- imbalanced-learn

## Dataset
The project uses a comprehensive credit card fraud dataset containing:
- **Training Data**: 335MB with transaction features and labels
- **Test Data**: 143MB for model evaluation
- **Features**: Transaction amount, time, location, merchant info, etc.
- **Target**: Binary classification (fraudulent/legitimate)

## Key Challenges Addressed
1. **Class Imbalance**: Fraudulent transactions are rare (~0.1%)
2. **Feature Engineering**: Creating meaningful features from raw data
3. **Model Selection**: Choosing algorithms that handle imbalanced data
4. **Performance Metrics**: Using appropriate metrics for fraud detection
5. **Real-time Processing**: Fast prediction for live transactions

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License. 