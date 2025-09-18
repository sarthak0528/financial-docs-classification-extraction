import joblib
from config.constants import MODELS_PATH

def load_classification_model():
    """Load TF-IDF vectorizer and Logistic Regression model."""
    vectorizer = joblib.load(f"{MODELS_PATH}/tfidf_vectorizer.pkl")
    model = joblib.load(f"{MODELS_PATH}/baseline_model.pkl")
    return vectorizer, model

def load_dl_model(model_path):
    """Load a deep learning Keras model."""
    from tensorflow.keras.models import load_model
    return load_model(model_path)