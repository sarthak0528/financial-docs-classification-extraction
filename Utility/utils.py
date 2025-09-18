import re
from bs4 import BeautifulSoup
from config.constants import MAX_LEN, VOCAB_SIZE


def clean_text(text):
    """Lowercase, remove unwanted characters, extra spaces."""
    try:
        text = text.lower()
        text = re.sub(r"[^a-z0-9.% ]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        print(f"[ERROR] Failed to clean text: {e}")
        return ""


def html_to_text(file_path):
    """Extract plain text from HTML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator=" ", strip=True)
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return ""
    except Exception as e:
        print(f"[ERROR] Failed to parse HTML file {file_path}: {e}")
        return ""


def chunk_text(text, chunk_size=300):
    """Split text into smaller chunks for RAG."""
    try:
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    except Exception as e:
        print(f"[ERROR] Failed to chunk text: {e}")
        return []
