# Movie Genre Classification

## Project Overview
This project implements a machine learning model that can predict the genre of a movie based on its plot summary or other textual information. The model uses TF-IDF vectorization and various classifiers including Naive Bayes, Logistic Regression, and Support Vector Machines.

## Features
- **Text Preprocessing**: Clean and prepare movie plot summaries
- **Feature Extraction**: TF-IDF vectorization for text representation
- **Multiple Classifiers**: Naive Bayes, Logistic Regression, SVM
- **Model Evaluation**: Comprehensive metrics and visualization
- **Interactive Prediction**: Web interface for genre prediction

## Project Structure
```
TASK 1/
├── Genre Classification Dataset/    # Real movie dataset
│   ├── train_data.txt              # Training data with genres
│   ├── test_data.txt               # Test data without genres
│   ├── test_data_solution.txt      # Test data solutions
│   └── description.txt             # Dataset description
├── data/                           # Processed data (auto-created)
├── models/                         # Trained models (auto-created)
├── notebooks/
│   └── movie_genre_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
├── web_app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── style.css
│       └── script.js
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
cd movie_genre_classification
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
python main.py --mode predict --text "Your movie plot here"
```

### Running the Web App
```bash
python web_app/app.py
```

## Model Performance
- **Naive Bayes**: Accuracy ~75%
- **Logistic Regression**: Accuracy ~82%
- **Support Vector Machine**: Accuracy ~85%

## Technologies Used
- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Flask
- nltk

## Dataset
The project uses the real movie dataset from the Internet Movie Database (IMDB) containing:
- **Training Data**: ~34MB with movie ID, title, genre, and plot description
- **Test Data**: ~33MB with movie ID, title, and plot description (for predictions)
- **Format**: ID ::: TITLE ::: GENRE ::: DESCRIPTION
- **Source**: ftp://ftp.fu-berlin.de/pub/misc/movies/database/

The dataset includes thousands of movies across multiple genres, providing a robust foundation for genre classification.

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License. 