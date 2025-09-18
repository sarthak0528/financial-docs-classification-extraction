# Base paths
DATA_PATH = r"C:\sarthak's dev\Intelligent_Document_Classifier_Extractor\data"
BALANCE_SHEETS_PATH = fr"{DATA_PATH}\Balance Sheets"
MODELS_PATH = r"C:\sarthak's dev\Intelligent_Document_Classifier_Extractor\Models"

# Preprocessing constants
VOCAB_SIZE = 20000
MAX_LEN = 512  # Will be adjusted dynamically in training
EMBED_DIM = 128

# Training constants
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 15

# TF-IDF
MAX_FEATURES = 5000

# Embeddings / Retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300
TOP_K = 3

# File names
CLASSIFICATION_MODEL_FILE = "baseline_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"