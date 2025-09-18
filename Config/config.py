import os
import joblib
from config.constants import (
    MODELS_PATH,
    CLASSIFICATION_MODEL_FILE,
    VECTORIZER_FILE,
    EMBEDDING_MODEL_NAME
)
from sentence_transformers import SentenceTransformer


# Load serialized models
def load_tfidf_and_model():
    """
    Load TF-IDF vectorizer and classification model.
    """
    try:
        vectorizer_path = os.path.join(MODELS_PATH, VECTORIZER_FILE)
        model_path = os.path.join(MODELS_PATH, CLASSIFICATION_MODEL_FILE)

        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)

        return vectorizer, model

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Unexpected issue while loading TF-IDF/model: {e}")
        return None, None


# Load embedding model
def load_embedding_model():
    """
    Load sentence transformer for embeddings.
    """
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] Unexpected issue while loading embedding model: {e}")
        return None


# One-time loads
try:
    VECTORIZER, CLASSIFICATION_MODEL = load_tfidf_and_model()
except Exception as e:
    print(f"[Warning] Could not load classification model/vectorizer: {e}")
    VECTORIZER, CLASSIFICATION_MODEL = None, None

try:
    EMBEDDING_MODEL = load_embedding_model()
except Exception as e:
    print(f"[Warning] Could not load embedding model: {e}")
    EMBEDDING_MODEL = None