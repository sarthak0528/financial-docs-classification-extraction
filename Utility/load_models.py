import joblib
from config.constants import MODELS_PATH


def load_classification_model():
    """Load TF-IDF vectorizer and Logistic Regression model."""
    try:
        vectorizer = joblib.load(f"{MODELS_PATH}/tfidf_vectorizer.pkl")
        model = joblib.load(f"{MODELS_PATH}/baseline_model.pkl")
        return vectorizer, model
    except FileNotFoundError as e:
        print(f"[ERROR] Model files not found: {e}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected issue while loading classification model: {e}")
        return None, None


def load_dl_model(model_path):
    """Load a deep learning Keras model."""
    try:
        from tensorflow.keras.models import load_model
        return load_model(model_path)
    except OSError as e:
        print(f"[ERROR] Could not load DL model from {model_path}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected issue while loading DL model: {e}")
        return None