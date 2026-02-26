# 📧 Spam Email Detection using Machine Learning

A complete machine learning project for detecting spam emails/SMS messages using Python.

## 🎯 Project Overview

This project implements a spam detection system using:
- **Naive Bayes** classifier
- **Logistic Regression** classifier

The models are trained on the SMS Spam Collection dataset and achieve high accuracy in distinguishing spam from legitimate messages.

## 📁 Project Structure

```
spam/
├── spam_detection.py       # Main Python script with full implementation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher

### Installation

1. Clone or download this repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script:
   ```bash
   python spam_detection.py
   ```

## 📊 Features

- **Data Loading**: Automatically loads the SMS Spam Collection dataset
- **Data Preprocessing**: Text cleaning, tokenization, TF-IDF vectorization
- **Model Training**: Trains Naive Bayes and Logistic Regression models
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report
- **Visualization**: Confusion matrix plots, model comparison charts
- **Prediction Function**: `predict_email(text)` for custom predictions

## 📈 Results

Both models achieve excellent accuracy (typically >95%) on the test set. The script compares their performance and identifies the best model.

## 🔮 Usage

Run the script directly or import the functions:

```python
from spam_detection import predict_email, main

# Run full pipeline
best_model, vectorizer, nb_model, lr_model = main()

# Predict a message
result = predict_email("Congratulations! You've won $1000!", best_model, vectorizer)
print(result['prediction'])  # Output: SPAM 🚫
```

## 📚 Dataset

The SMS Spam Collection dataset contains 5,574 SMS messages tagged as spam or ham. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

## 📝 License

This project is open source and available for educational purposes.
