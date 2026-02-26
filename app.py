"""
📧 Spam Email Detection - Web Application
==========================================
A Flask web app for spam detection using trained ML models.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

app = Flask(__name__)

# =============================================================================
# STOPWORDS & TEXT CLEANING
# =============================================================================

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}


def clean_text(text):
    """Clean text for prediction."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    return ' '.join(tokens)


# =============================================================================
# LOAD AND TRAIN MODEL
# =============================================================================

print("🔄 Loading dataset and training model...")

# Try to load local dataset first, otherwise use online
local_dataset = os.path.join(os.path.dirname(__file__), 'dataset.csv')
if os.path.exists(local_dataset):
    df = pd.read_csv(local_dataset)
    print(f"✅ Loaded local dataset: {local_dataset}")
else:
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    print(f"✅ Loaded online dataset")

# Store original dataframe for dataset exploration
original_df = df.copy()

# Preprocess
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
df['cleaned_message'] = df['message'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label_encoded']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))

# Use best model
model = nb_model if nb_accuracy >= lr_accuracy else lr_model
accuracy = max(nb_accuracy, lr_accuracy)
best_model_name = "Naive Bayes" if nb_accuracy >= lr_accuracy else "Logistic Regression"

print(f"✅ Models trained!")
print(f"   Naive Bayes: {nb_accuracy*100:.2f}%")
print(f"   Logistic Regression: {lr_accuracy*100:.2f}%")
print(f"   Best Model: {best_model_name}")


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html', accuracy=f"{accuracy*100:.2f}")


@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a message is spam or ham."""
    try:
        # Get message from request
        data = request.get_json()
        message = data.get('message', '')
        
        if not message.strip():
            return jsonify({'error': 'Please enter a message'}), 400
        
        # Clean and vectorize
        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
        
        result = {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': round(confidence, 2),
            'cleaned_text': cleaned,
            'is_spam': bool(prediction)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get model statistics."""
    return jsonify({
        'accuracy': round(accuracy * 100, 2),
        'nb_accuracy': round(nb_accuracy * 100, 2),
        'lr_accuracy': round(lr_accuracy * 100, 2),
        'best_model': best_model_name,
        'total_samples': len(df),
        'spam_count': int(df['label_encoded'].sum()),
        'ham_count': int(len(df) - df['label_encoded'].sum()),
        'features': vectorizer.max_features
    })


@app.route('/dataset')
def get_dataset():
    """Get dataset samples with optional filtering."""
    filter_type = request.args.get('filter', 'all')  # all, spam, ham
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    # Filter data
    if filter_type == 'spam':
        filtered_df = original_df[original_df['label'] == 'spam']
    elif filter_type == 'ham':
        filtered_df = original_df[original_df['label'] == 'ham']
    else:
        filtered_df = original_df
    
    # Pagination
    total = len(filtered_df)
    start = (page - 1) * per_page
    end = start + per_page
    
    data = filtered_df.iloc[start:end].to_dict('records')
    
    return jsonify({
        'data': data,
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'spam_count': len(original_df[original_df['label'] == 'spam']),
        'ham_count': len(original_df[original_df['label'] == 'ham'])
    })


@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    """Analyze all messages in dataset with predictions."""
    results = []
    
    for idx, row in original_df.iterrows():
        cleaned = clean_text(row['message'])
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = max(probabilities) * 100
        
        actual = 1 if row['label'] == 'spam' else 0
        is_correct = prediction == actual
        
        results.append({
            'message': row['message'][:100] + ('...' if len(row['message']) > 100 else ''),
            'actual': row['label'],
            'predicted': 'spam' if prediction == 1 else 'ham',
            'confidence': round(confidence, 2),
            'correct': is_correct
        })
    
    # Calculate metrics
    correct_count = sum(1 for r in results if r['correct'])
    accuracy_pct = (correct_count / len(results)) * 100
    
    return jsonify({
        'results': results[:50],  # Return first 50 for performance
        'total_analyzed': len(results),
        'correct_predictions': correct_count,
        'accuracy': round(accuracy_pct, 2)
    })


@app.route('/predict_sample', methods=['POST'])
def predict_sample():
    """Predict a sample from the dataset by index."""
    data = request.get_json()
    index = data.get('index', 0)
    
    if index < 0 or index >= len(original_df):
        return jsonify({'error': 'Invalid index'}), 400
    
    row = original_df.iloc[index]
    message = row['message']
    actual_label = row['label']
    
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = max(probabilities) * 100
    
    return jsonify({
        'message': message,
        'actual': actual_label,
        'predicted': 'spam' if prediction == 1 else 'ham',
        'confidence': round(confidence, 2),
        'correct': (prediction == 1 and actual_label == 'spam') or (prediction == 0 and actual_label == 'ham')
    })


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Spam Detection Web App...")
    print("📍 Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
