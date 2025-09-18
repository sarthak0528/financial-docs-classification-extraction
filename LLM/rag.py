"""
Retrieval (FAISS) + simple local answerer for financial QA (no external API required).

Usage:
    from LLM.rag import RAGPipeline, rag_qa

    # Prepare documents = [{"file": "a.html", "text": "..."} , ...]
    rag = RAGPipeline()
    rag.build_index(documents)           # build in-memory index
    # or rag.load_index() to restore from disk if previously saved

    answers = rag_qa(rag, "What is the Total Assets?", top_k=3)
    print(answers)
"""

import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from config.constants import MODELS_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, TOP_K

# Where we persist index and chunks
_INDEX_FILE = os.path.join(MODELS_PATH, "faiss_index.bin")
_CHUNKS_FILE = os.path.join(MODELS_PATH, "faiss_chunks.json")

# Regex to capture common financial number formats like 42,389.09 or 439,697
_NUM_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")


class RAGPipeline:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            # Sentence-transformer model for embeddings
            self.embedder = SentenceTransformer(model_name)
            self.index = None
            self.chunks = []
        except Exception as e:
            print(f"[ERROR] Failed to initialize embedder: {e}")
            self.embedder, self.index, self.chunks = None, None, []

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
        try:
            words = text.split()
            return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        except Exception as e:
            print(f"[ERROR] Failed to chunk text: {e}")
            return []

    def build_index(self, documents: list, chunk_size: int = CHUNK_SIZE, persist: bool = True):
        try:
            # prepare chunks
            self.chunks = []
            for doc in documents:
                text = doc.get("text", "")
                file = doc.get("file", "unknown")
                for chunk in self._chunk_text(text, chunk_size=chunk_size):
                    self.chunks.append({"file": file, "chunk": chunk})

            if not self.chunks:
                raise ValueError("No chunks created. Are your documents empty?")

            if self.embedder is None:
                raise RuntimeError("Embedder not initialized.")

            # create embeddings in batches
            chunk_texts = [c["chunk"] for c in self.chunks]
            embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

            # create FAISS index (flat L2)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.asarray(embeddings).astype('float32'))

            if persist:
                self.save_index()

            print(f"[RAG] Built FAISS index with {len(self.chunks)} chunks (dim={dim})")

        except Exception as e:
            print(f"[ERROR] Failed to build index: {e}")
            self.index, self.chunks = None, []

    def save_index(self):
        """Persist FAISS index and chunks to disk (MODELS_PATH)"""
        try:
            os.makedirs(MODELS_PATH, exist_ok=True)
            if self.index is None:
                raise RuntimeError("No index to save. Build the index first.")
            faiss.write_index(self.index, _INDEX_FILE)
            with open(_CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            print(f"[RAG] Index saved to {_INDEX_FILE} and chunks to {_CHUNKS_FILE}")
        except Exception as e:
            print(f"[ERROR] Failed to save index: {e}")

    def load_index(self):
        """Load persisted FAISS index and chunks (if present)"""
        try:
            if not os.path.exists(_INDEX_FILE) or not os.path.exists(_CHUNKS_FILE):
                raise FileNotFoundError("Persisted index or chunks not found in Models folder.")
            self.index = faiss.read_index(_INDEX_FILE)
            with open(_CHUNKS_FILE, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[RAG] Loaded index ({len(self.chunks)} chunks) from disk.")
        except Exception as e:
            print(f"[ERROR] Failed to load index: {e}")
            self.index, self.chunks = None, []

    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Return top_k most similar chunks (list of dicts: {"file","chunk"})
        """
        try:
            if self.index is None:
                raise RuntimeError("Index not built or loaded. Call build_index() or load_index().")

            if self.embedder is None:
                raise RuntimeError("Embedder not initialized.")

            q_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
            distances, indices = self.index.search(q_emb, top_k)
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.chunks):
                    results.append(self.chunks[idx])
            return results
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return []


# -------------------------
# Simple local "LLM-like" extractor (no API)
# -------------------------
def _extract_numeric_for_key(keyword: str, text: str):
    try:
        results = []
        for m in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
            start = m.start()
            snippet = text[start:start + 200]
            matches = _NUM_PATTERN.findall(snippet)
            if matches:
                normalized = []
                for num in matches:
                    clean = num.replace(",", "")
                    try:
                        val = float(clean)
                    except Exception:
                        continue
                    normalized.append((num, val))
                results.extend(normalized)

        if not results:
            all_matches = _NUM_PATTERN.findall(text)
            normalized = []
            for num in all_matches:
                try:
                    val = float(num.replace(",", ""))
                except Exception:
                    continue
                normalized.append((num, val))
            if normalized:
                best = max(normalized, key=lambda x: x[1])
                return best[0]
            return None

        best = max(results, key=lambda x: x[1])
        return best[0]

    except Exception as e:
        print(f"[ERROR] Failed numeric extraction for {keyword}: {e}")
        return None


def query_llm_mock(context: str, question: str):
    try:
        q = question.lower()
        if "current asset" in q:
            val = _extract_numeric_for_key("current asset", context) or _extract_numeric_for_key("current", context)
            return val or ("[Not found] " + context[:300])
        if "total asset" in q or "total assets" in q:
            val = _extract_numeric_for_key("total asset", context) or _extract_numeric_for_key("total", context)
            return val or ("[Not found] " + context[:300])
        if "equity" in q or "share capital" in q or "net worth" in q:
            val = _extract_numeric_for_key("equity", context) or _extract_numeric_for_key("share capital", context)
            return val or ("[Not found] " + context[:300])

        return context[:500]
    except Exception as e:
        print(f"[ERROR] Mock query failed: {e}")
        return "[Error] Could not answer."


def rag_qa(rag_pipeline: RAGPipeline, question: str, top_k: int = TOP_K):
    try:
        chunks = rag_pipeline.retrieve(question, top_k=top_k)
        if not chunks:
            return "[No relevant context found]"

        context = "\n\n".join([c["chunk"] for c in chunks])
        answer = query_llm_mock(context, question)
        return answer
    except Exception as e:
        print(f"[ERROR] RAG QA failed: {e}")
        return "[Error] QA failed."


# -------------------------
# Simple demo when run as script
# -------------------------
if __name__ == "__main__":
    try:
        docs = [
            {"file": "bs1.html", "text": "Balance Sheet. Current Assets 32,518.63 Total Assets 41,973.88 Equity 25,813.15"},
            {"file": "bs2.html", "text": "Some company. Total Assets 190,096.93 Current Assets 82,249.03 Equity 165,598.09"}
        ]

        rag = RAGPipeline()
        rag.build_index(docs, chunk_size=50, persist=False)

        print("Q:", "What is the value of Current Assets?")
        print("A:", rag_qa(rag, "What is the value of Current Assets?", top_k=2))
        print()
        print("Q:", "What is the Total Assets amount?")
        print("A:", rag_qa(rag, "What is the Total Assets amount?", top_k=2))
        print()
        print("Q:", "How much is Equity?")
        print("A:", rag_qa(rag, "How much is Equity?", top_k=2))

    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")