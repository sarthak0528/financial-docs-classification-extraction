import joblib
import os
from config.constants import MODELS_PATH

def load_model():
    model_path = os.path.join(MODELS_PATH, "baseline_model.pkl")
    vec_path = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("Model or vectorizer not found. Run train.py first.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer

def predict(text: str):
    model, vectorizer = load_model()
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

if __name__ == "__main__":
    sample_text = "This financial report includes details of current assets and liabilities."
    label = predict(sample_text)
    print(f"Predicted label: {label}")