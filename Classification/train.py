import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from config.constants import DATA_PATH, MODELS_PATH, MAX_FEATURES, TEST_SIZE, RANDOM_STATE

def load_data():
    """Load preprocessed CSV data (assume you already created df somewhere)."""
    file_path = os.path.join(DATA_PATH, "preprocessed_docs.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Expected data at {file_path}, but not found.")
    return pd.read_csv(file_path)

def train_model():
    df = load_data()

    X = df["clean_text"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    # Save models
    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_PATH, "baseline_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl"))

    acc = model.score(X_test_vec, y_test)
    print(f"Model trained. Test Accuracy = {acc:.4f}")

if __name__ == "__main__":
    train_model()