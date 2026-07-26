import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
print("=" * 60)
print("STEAM REVIEW SENTIMENT MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("data/cleaned_games.csv")

print("\nDataset Loaded Successfully!")
print(f"Total Reviews : {len(df):,}")
print("\n" + "=" * 60)
print("PREPARING DATA")
print("=" * 60)

X = df["review"]
y = df["voted_up"]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train):,}")
print(f"Testing Samples  : {len(X_test):,}")
print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training Matrix Shape : {X_train_tfidf.shape}")
print(f"Testing Matrix Shape  : {X_test_tfidf.shape}")
print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("=" * 60)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Training Complete!")
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("Model saved successfully!")
print("Vectorizer saved successfully!")
print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
