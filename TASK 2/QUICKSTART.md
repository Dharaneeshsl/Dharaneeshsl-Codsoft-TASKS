# Credit Card Fraud Detection - Quick Start Guide

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd "TASK 2"
pip install -r requirements.txt
```

### 2. Test Dataset Loading
```bash
python test_dataset.py
```
This will:
- Verify dataset files are present
- Load and explore the data
- Test the preprocessing pipeline
- Show data statistics

### 3. Run the Demo
```bash
python demo.py
```
This will:
- Load and preprocess a sample of the data
- Train multiple ML models
- Evaluate model performance
- Test sample predictions
- Show feature importance

### 4. Train Full Models
```bash
python main.py --mode train
```
This will:
- Load the complete dataset
- Train all models (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost)
- Save trained models to `models/` directory
- Generate evaluation reports

### 5. Start Web Interface
```bash
python web_app/app.py
```
Then open your browser to `http://localhost:5000`

## 📁 Project Structure
```
TASK 2/
├── archive (4)/                    # Original dataset
│   ├── fraudTrain.csv             # Training data (335MB)
│   └── fraudTest.csv              # Test data (143MB)
├── src/                          # Core modules
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── model_training.py         # Model training
│   ├── evaluation.py             # Model evaluation
│   └── prediction.py             # Real-time predictions
├── web_app/                      # Flask web interface
│   ├── app.py                    # Web server
│   ├── templates/index.html      # Web UI
│   └── static/                   # CSS and JS files
├── models/                       # Trained models (auto-created)
├── main.py                       # Command-line interface
├── demo.py                       # Demo script
├── test_dataset.py               # Dataset testing
└── requirements.txt              # Dependencies
```

## 🎯 Key Features

### Data Preprocessing
- **Feature Engineering**: Time-based, amount-based, location-based features
- **Class Imbalance Handling**: SMOTE oversampling
- **Feature Selection**: SelectKBest with F-statistic
- **Data Scaling**: StandardScaler normalization

### Machine Learning Models
- **Logistic Regression**: Baseline model with balanced class weights
- **Decision Tree**: Interpretable model with balanced class weights
- **Random Forest**: Ensemble model with 100 estimators
- **Support Vector Machine**: RBF kernel with balanced class weights
- **XGBoost**: Gradient boosting with class imbalance handling

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

### Web Interface
- **Real-time Analysis**: Single transaction analysis
- **Batch Processing**: CSV file upload and analysis
- **Model Selection**: Choose from different ML models
- **Visual Results**: Interactive charts and tables
- **Risk Assessment**: Confidence scores and recommendations

## 🔧 Usage Examples

### Command Line Prediction
```bash
# Single transaction prediction
python main.py --mode predict --data '{"amt": 150.0, "lat": 40.7128, "long": -74.0060}'

# Evaluate existing models
python main.py --mode evaluate
```

### Python API
```python
from src.prediction import FraudPredictor

# Initialize predictor
predictor = FraudPredictor()

# Single prediction
transaction = {
    'amt': 150.0,
    'lat': 40.7128,
    'long': -74.0060,
    'merchant': 'online',
    'hour': 14,
    'day_of_week': 2
}

prediction, confidence = predictor.predict_fraud(transaction)
print(f"Fraudulent: {prediction}, Confidence: {confidence:.3f}")

# Comprehensive analysis
analysis = predictor.analyze_transaction(transaction)
print(f"Risk Level: {analysis['risk_level']}")
print(f"Recommendations: {analysis['recommendations']}")
```

## 📊 Expected Performance

Based on the credit card fraud dataset:
- **Random Forest**: ~99.9% accuracy, ~95% precision
- **XGBoost**: ~99.9% accuracy, ~96% precision
- **Logistic Regression**: ~99.8% accuracy, ~92% precision
- **Decision Tree**: ~99.7% accuracy, ~90% precision

## 🚨 Important Notes

1. **Dataset Size**: The full dataset is ~478MB total. For faster testing, use the demo script which samples 10k records.

2. **Class Imbalance**: Fraudulent transactions are rare (~0.1%). The system uses SMOTE to handle this imbalance.

3. **Model Training**: Full training may take 10-30 minutes depending on your hardware.

4. **Memory Requirements**: Ensure you have at least 4GB RAM available for full dataset processing.

5. **Web Interface**: Requires trained models to be present in the `models/` directory.

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

2. **Memory Error**: Use the demo script or reduce sample size in `main.py`

3. **Model Files Not Found**: Run `python main.py --mode train` first

4. **Web Interface Error**: Ensure models are trained and Flask is installed

### Performance Tips

1. **Use SSD**: Dataset loading is faster with solid-state drives
2. **Increase RAM**: More memory allows larger batch processing
3. **Use GPU**: XGBoost can utilize GPU acceleration if available
4. **Sample Data**: Use smaller samples for development and testing

## 📈 Next Steps

1. **Hyperparameter Tuning**: Use GridSearchCV for model optimization
2. **Feature Engineering**: Add domain-specific features
3. **Model Ensemble**: Combine multiple models for better performance
4. **Real-time Integration**: Connect to live transaction streams
5. **API Deployment**: Deploy as a REST API service

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 