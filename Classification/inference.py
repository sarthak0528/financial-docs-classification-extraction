import joblib
import os
from config.constants import MODELS_PATH


def load_model():
    try:
        model_path = os.path.join(MODELS_PATH, "baseline_model.pkl")
        vec_path = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path) or not os.path.exists(vec_path):
            raise FileNotFoundError("Model or vectorizer not found. Run train.py first.")

        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected issue while loading model/vectorizer: {e}")
        return None, None


def predict(text: str):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        model, vectorizer = load_model()
        if model is None or vectorizer is None:
            raise RuntimeError("Model or vectorizer could not be loaded.")

        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        return prediction

    except ValueError as ve:
        print(f"[ERROR] {ve}")
        return None
    except RuntimeError as re:
        print(f"[ERROR] {re}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected issue during prediction: {e}")
        return None


if __name__ == "__main__":
    try:
        sample_text = "This financial report includes details of current assets and liabilities."
        label = predict(sample_text)
        if label is not None:
            print(f"Predicted label: {label}")
        else:
            print("Prediction could not be made.")
    except Exception as e:
        print(f"[ERROR] Failed to run inference: {e}")