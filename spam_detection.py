"""
📧 Spam Email Detection using Machine Learning
================================================
A complete machine learning project for detecting spam emails/SMS messages.

Models: Naive Bayes & Logistic Regression
Dataset: SMS Spam Collection Dataset
"""

# =============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# English stopwords list (common words that don't carry meaning)
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

print("✅ All libraries imported successfully!\n")


# =============================================================================
# 2. LOAD THE DATASET
# =============================================================================

def load_dataset():
    """Load the SMS Spam Collection dataset from URL."""
    print("=" * 60)
    print("📊 LOADING DATASET")
    print("=" * 60)
    
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(df)}")
    
    # Display first 5 rows
    print("\n📝 First 5 rows of the dataset:")
    print(df.head())
    
    # Dataset info
    print("\n📈 Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    class_dist = df['label'].value_counts()
    print(class_dist)
    print(f"\nSpam percentage: {(class_dist['spam']/len(df))*100:.2f}%")
    print(f"Ham percentage: {(class_dist['ham']/len(df))*100:.2f}%")
    
    return df


def plot_class_distribution(df):
    """Visualize class distribution."""
    class_dist = df['label'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = ['#2ecc71', '#e74c3c']
    
    # Bar plot
    axes[0].bar(class_dist.index, class_dist.values, color=colors)
    axes[0].set_title('Class Distribution (Bar Chart)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(class_dist.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%',
                colors=colors, explode=(0, 0.1), shadow=True, startangle=90)
    axes[1].set_title('Class Distribution (Pie Chart)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Class distribution chart saved as 'class_distribution.png'")


# =============================================================================
# 3. DATA PREPROCESSING
# =============================================================================

def clean_text(text):
    """
    Clean text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing numbers
    - Removing stopwords
    - Tokenization
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Simple tokenization (split by whitespace)
    tokens = text.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in STOPWORDS]
    
    # Join tokens back to string
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


def preprocess_data(df):
    """Perform all preprocessing steps."""
    print("\n" + "=" * 60)
    print("🔧 DATA PREPROCESSING")
    print("=" * 60)
    
    # Convert labels to binary (spam=1, ham=0)
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    print("✅ Labels converted to binary (spam=1, ham=0)")
    
    # Clean text
    print("🔄 Cleaning text data... This may take a moment.")
    df['cleaned_message'] = df['message'].apply(clean_text)
    print("✅ Text cleaning completed!")
    
    # Show before and after
    print("\n📝 Sample - Before and After Cleaning:")
    print("-" * 70)
    for i in range(3):
        print(f"[Original {i+1}]: {df['message'].iloc[i][:60]}...")
        print(f"[Cleaned {i+1}]:  {df['cleaned_message'].iloc[i][:60]}...")
        print()
    
    return df


def vectorize_data(df):
    """Apply TF-IDF Vectorization."""
    print("🔄 Applying TF-IDF Vectorization...")
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
    y = df['label_encoded']
    
    print(f"✅ TF-IDF Vectorization completed!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features (words): {X.shape[1]}")
    
    return X, y, tfidf_vectorizer


# =============================================================================
# 4. SPLIT DATASET
# =============================================================================

def split_data(X, y):
    """Split data into training and testing sets (80-20)."""
    print("\n" + "=" * 60)
    print("📊 SPLITTING DATASET")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"✅ Dataset split completed!")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# 5. TRAIN MODELS
# =============================================================================

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model."""
    print("\n🔄 Training Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    print("✅ Naive Bayes model trained!")
    return nb_model


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    print("🔄 Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print("✅ Logistic Regression model trained!")
    return lr_model


# =============================================================================
# 6. EVALUATE MODELS
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and print metrics."""
    print("\n" + "=" * 60)
    print(f"📊 {model_name.upper()} MODEL EVALUATION")
    print("=" * 60)
    
    predictions = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    print(f"\n📈 Confusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
    
    return accuracy, cm, predictions


# =============================================================================
# 7. PLOT CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrices(nb_cm, lr_cm):
    """Plot confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Naive Bayes
    sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                annot_kws={'size': 14})
    axes[0].set_title('Naive Bayes - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Logistic Regression
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                annot_kws={'size': 14})
    axes[1].set_title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Confusion matrices saved as 'confusion_matrices.png'")


# =============================================================================
# 8. COMPARE MODELS
# =============================================================================

def compare_models(nb_accuracy, lr_accuracy):
    """Compare and visualize model performances."""
    print("\n" + "=" * 60)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Accuracy':<15} {'Accuracy (%)':<15}")
    print("-" * 55)
    print(f"{'Naive Bayes':<25} {nb_accuracy:<15.4f} {nb_accuracy*100:.2f}%")
    print(f"{'Logistic Regression':<25} {lr_accuracy:<15.4f} {lr_accuracy*100:.2f}%")
    
    # Determine winner
    print("\n" + "=" * 60)
    if nb_accuracy > lr_accuracy:
        print(f"🏆 WINNER: Naive Bayes with {nb_accuracy*100:.2f}% accuracy!")
        print(f"   Naive Bayes outperforms Logistic Regression by {(nb_accuracy - lr_accuracy)*100:.2f}%")
        best_model_name = "Naive Bayes"
    elif lr_accuracy > nb_accuracy:
        print(f"🏆 WINNER: Logistic Regression with {lr_accuracy*100:.2f}% accuracy!")
        print(f"   Logistic Regression outperforms Naive Bayes by {(lr_accuracy - nb_accuracy)*100:.2f}%")
        best_model_name = "Logistic Regression"
    else:
        print(f"🤝 TIE: Both models have equal accuracy of {nb_accuracy*100:.2f}%!")
        best_model_name = "Naive Bayes"
    print("=" * 60)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    models = ['Naive Bayes', 'Logistic Regression']
    accuracies = [nb_accuracy * 100, lr_accuracy * 100]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.7, label='95% threshold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Model comparison chart saved as 'model_comparison.png'")
    
    return best_model_name


# =============================================================================
# 9. PREDICTION FUNCTION
# =============================================================================

def predict_email(text, model, vectorizer):
    """
    Predict whether an email/message is spam or not.
    
    Parameters:
    -----------
    text : str
        The email or message text to classify
    model : sklearn model
        The trained model to use
    vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer
    
    Returns:
    --------
    dict : Contains prediction result and confidence
    """
    # Clean the input text
    cleaned_text = clean_text(text)
    
    # Vectorize the cleaned text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get prediction probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vectorized)[0]
        confidence = max(probabilities) * 100
    else:
        confidence = None
    
    result = {
        'input_text': text,
        'cleaned_text': cleaned_text,
        'prediction': 'SPAM 🚫' if prediction == 1 else 'HAM (Not Spam) ✅',
        'is_spam': bool(prediction),
        'confidence': f"{confidence:.2f}%" if confidence else "N/A"
    }
    
    return result


def test_predictions(model, vectorizer):
    """Test the prediction function with sample messages."""
    print("\n" + "=" * 70)
    print("📧 TESTING PREDICTION FUNCTION")
    print("=" * 70)
    
    test_messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been compromised. Send your password immediately!",
        "Hi Mom, I'll be home by 6pm for dinner.",
        "FREE VIAGRA! LIMITED TIME OFFER! BUY NOW AND SAVE 90%!!!",
        "The project meeting has been rescheduled to Friday at 2pm."
    ]
    
    for i, msg in enumerate(test_messages, 1):
        result = predict_email(msg, model, vectorizer)
        print(f"\n📨 Message {i}:")
        print(f"   Text: \"{msg[:60]}{'...' if len(msg) > 60 else ''}\"")
        print(f"   ➡️  Prediction: {result['prediction']}")
        print(f"   📊 Confidence: {result['confidence']}")


# =============================================================================
# 10. PRINT SUMMARY
# =============================================================================

def print_summary(df, y_train, y_test, nb_accuracy, lr_accuracy, best_model_name):
    """Print final project summary."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "📊 PROJECT SUMMARY" + " " * 29 + "║")
    print("╠" + "=" * 68 + "╣")
    print(f"║  Dataset Size: {len(df):,} messages" + " " * (51 - len(str(len(df)))) + "║")
    print(f"║  Training Set: {len(y_train):,} messages ({len(y_train)/len(df)*100:.1f}%)" + " " * (36 - len(str(len(y_train)))) + "║")
    print(f"║  Testing Set:  {len(y_test):,} messages ({len(y_test)/len(df)*100:.1f}%)" + " " * (37 - len(str(len(y_test)))) + "║")
    print("╠" + "=" * 68 + "╣")
    print("║  MODEL PERFORMANCE:" + " " * 48 + "║")
    print(f"║    • Naive Bayes:         {nb_accuracy*100:.2f}% accuracy" + " " * 27 + "║")
    print(f"║    • Logistic Regression: {lr_accuracy*100:.2f}% accuracy" + " " * 27 + "║")
    print("╠" + "=" * 68 + "╣")
    print(f"║  🏆 Best Model: {best_model_name}" + " " * (51 - len(best_model_name)) + "║")
    print("╚" + "=" * 68 + "╝")


# =============================================================================
# 11. HOW THE MODEL WORKS
# =============================================================================

def print_explanation():
    """Print explanation of how the model works."""
    explanation = """
╔══════════════════════════════════════════════════════════════════════╗
║           🔍 HOW THE MODEL WORKS (Simple Explanation)                ║
╚══════════════════════════════════════════════════════════════════════╝

📌 WHAT IS THE GOAL?
   Our goal is to teach a computer to automatically identify whether an 
   email or text message is SPAM (unwanted) or HAM (legitimate).

📌 STEP-BY-STEP PROCESS:

   1. DATA COLLECTION 📚
      - We gathered thousands of messages labeled as "spam" or "ham"
      - Like showing a child examples of apples vs oranges

   2. TEXT CLEANING 🧹
      - Made everything lowercase ("HELLO" → "hello")
      - Removed punctuation (!@#$%)
      - Removed common words ("the", "is", "and")

   3. CONVERTING TEXT TO NUMBERS (TF-IDF) 🔢
      - Computers only understand numbers
      - TF-IDF counts word frequency and importance
      - Rare words like "FREE" get more weight than "the"

   4. TRAINING THE MODELS 🎓
      - Naive Bayes: Uses probability theory
      - Logistic Regression: Draws a mathematical line to separate classes

   5. MAKING PREDICTIONS 🔮
      - Clean new message → Convert to numbers → Model predicts SPAM/HAM

📌 WHY DOES IT WORK?
   Spam messages have patterns:
   - Words like "FREE", "WINNER", "CLICK", "URGENT"
   - Excessive punctuation (!!!)
   - ALL CAPS

📌 REAL-WORLD ANALOGY 💡
   Like a mail sorter who has seen millions of letters:
   - "YOU'VE WON A PRIZE!" = junk mail
   - "Work Meeting Tomorrow" = important
   Our model does this mathematically and much faster!

══════════════════════════════════════════════════════════════════════════
"""
    print(explanation)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the entire spam detection pipeline."""
    print("\n" + "=" * 70)
    print("       📧 SPAM EMAIL DETECTION USING MACHINE LEARNING")
    print("=" * 70)
    
    # Step 1: Load dataset
    df = load_dataset()
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Step 2: Preprocess data
    df = preprocess_data(df)
    
    # Step 3: Vectorize data
    X, y, tfidf_vectorizer = vectorize_data(df)
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 5: Train models
    print("\n" + "=" * 60)
    print("🎓 TRAINING MODELS")
    print("=" * 60)
    nb_model = train_naive_bayes(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Step 6: Evaluate models
    nb_accuracy, nb_cm, _ = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    lr_accuracy, lr_cm, _ = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Step 7: Plot confusion matrices
    plot_confusion_matrices(nb_cm, lr_cm)
    
    # Step 8: Compare models
    best_model_name = compare_models(nb_accuracy, lr_accuracy)
    
    # Select best model
    best_model = nb_model if best_model_name == "Naive Bayes" else lr_model
    
    # Step 9: Test predictions
    test_predictions(best_model, tfidf_vectorizer)
    
    # Step 10: Print summary
    print_summary(df, y_train, y_test, nb_accuracy, lr_accuracy, best_model_name)
    
    # Step 11: Print explanation
    print_explanation()
    
    print("\n🎉 PROJECT COMPLETE!")
    print("=" * 70)
    
    # Return models and vectorizer for interactive use
    return best_model, tfidf_vectorizer, nb_model, lr_model


# =============================================================================
# RUN THE PROJECT
# =============================================================================

if __name__ == "__main__":
    # Run the main pipeline
    best_model, vectorizer, nb_model, lr_model = main()
    
    # Interactive prediction example
    print("\n" + "=" * 70)
    print("🔮 TRY YOUR OWN MESSAGE")
    print("=" * 70)
    
    your_message = "Win a brand new iPhone 15! Just text WIN to 12345!"
    result = predict_email(your_message, best_model, vectorizer)
    
    print(f"\n📧 Your Message: \"{your_message}\"")
    print(f"\n🔍 Analysis:")
    print(f"   - Cleaned Text: {result['cleaned_text']}")
    print(f"   - Prediction: {result['prediction']}")
    print(f"   - Confidence: {result['confidence']}")
