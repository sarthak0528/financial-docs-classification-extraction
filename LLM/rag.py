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
        # Sentence-transformer model for embeddings
        self.embedder = SentenceTransformer(model_name)
        self.index = None               # FAISS index
        self.chunks = []                # list of {"file":..., "chunk":...}

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_index(self, documents: list, chunk_size: int = CHUNK_SIZE, persist: bool = True):
        """
        Build FAISS index from list of documents.
        documents: list of {"file": filename, "text": fulltext}
        persist: if True, save index and chunks to MODELS_PATH
        """
        # prepare chunks
        self.chunks = []
        for doc in documents:
            text = doc.get("text", "")
            file = doc.get("file", "unknown")
            for chunk in self._chunk_text(text, chunk_size=chunk_size):
                self.chunks.append({"file": file, "chunk": chunk})

        if not self.chunks:
            raise ValueError("No chunks created. Are your documents empty?")

        # create embeddings in batches for memory safety
        chunk_texts = [c["chunk"] for c in self.chunks]
        embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True)

        # create FAISS index (flat L2)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.asarray(embeddings).astype('float32'))

        if persist:
            self.save_index()

        print(f"[RAG] Built FAISS index with {len(self.chunks)} chunks (dim={dim})")

    def save_index(self):
        """Persist FAISS index and chunks to disk (MODELS_PATH)"""
        os.makedirs(MODELS_PATH, exist_ok=True)
        if self.index is None:
            raise RuntimeError("No index to save. Build the index first.")
        faiss.write_index(self.index, _INDEX_FILE)
        with open(_CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"[RAG] Index saved to {_INDEX_FILE} and chunks to {_CHUNKS_FILE}")

    def load_index(self):
        """Load persisted FAISS index and chunks (if present)"""
        if not os.path.exists(_INDEX_FILE) or not os.path.exists(_CHUNKS_FILE):
            raise FileNotFoundError("Persisted index or chunks not found in Models folder.")
        self.index = faiss.read_index(_INDEX_FILE)
        with open(_CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[RAG] Loaded index ({len(self.chunks)} chunks) from disk.")

    def retrieve(self, query: str, top_k: int = TOP_K):
        """
        Return top_k most similar chunks (list of dicts: {"file","chunk"})
        """
        if self.index is None:
            raise RuntimeError("Index not built or loaded. Call build_index() or load_index().")

        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(self.chunks[idx])
        return results


# -------------------------
# Simple local "LLM-like" extractor (no API)
# -------------------------
def _extract_numeric_for_key(keyword: str, text: str):
    """
    Given a keyword and a block of text, attempt to extract the most-likely numeric value nearby.
    Strategy:
      - find keyword occurrences, look 200 chars forward, capture number tokens with regex
      - choose the largest numeric value found (typical for 'Total' rows)
    Returns None if not found.
    """
    results = []
    for m in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE):
        start = m.start()
        snippet = text[start:start + 200]  # look ahead
        matches = _NUM_PATTERN.findall(snippet)
        if matches:
            # normalize matched numbers and keep them
            normalized = []
            for num in matches:
                # handle 15.393 (possible thousand-dot) vs decimals: ambiguous - keep as-is
                clean = num.replace(",", "")
                try:
                    val = float(clean)
                except:
                    continue
                normalized.append((num, val))
            results.extend(normalized)

    if not results:
        # fallback: search entire text for numbers
        all_matches = _NUM_PATTERN.findall(text)
        normalized = []
        for num in all_matches:
            try:
                val = float(num.replace(",", ""))
            except:
                continue
            normalized.append((num, val))
        if normalized:
            # return largest
            best = max(normalized, key=lambda x: x[1])
            return best[0]
        return None

    # pick candidate with largest numeric value (commonly the total)
    best = max(results, key=lambda x: x[1])
    return best[0]


def query_llm_mock(context: str, question: str):
    """
    Lightweight 'answerer' that tries to extract numeric financial fields from context.
    - If question mentions keywords like 'current assets', 'total assets', 'equity',
      it will search context chunks for nearby numbers and return them.
    - Otherwise, returns a short summary (first 300 chars) of context.
    """
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

    # fallback: return combined context (short)
    return context[:500]


def rag_qa(rag_pipeline: RAGPipeline, question: str, top_k: int = TOP_K):
    """
    End-to-end retrieval + local-answer pipeline:
      1) retrieve top_k chunks for question
      2) combine into context
      3) run lightweight local answerer to extract numbers or return short context
    Returns the answer (string).
    """
    chunks = rag_pipeline.retrieve(question, top_k=top_k)
    if not chunks:
        return "[No relevant context found]"

    context = "\n\n".join([c["chunk"] for c in chunks])
    answer = query_llm_mock(context, question)
    return answer


# -------------------------
# Simple demo when run as script
# -------------------------
if __name__ == "__main__":
    # minimal demo documents
    docs = [
        {"file": "bs1.html", "text": "Balance Sheet. Current Assets 32,518.63 Total Assets 41,973.88 Equity 25,813.15"},
        {"file": "bs2.html", "text": "Some company. Total Assets 190,096.93 Current Assets 82,249.03 Equity 165,598.09"}
    ]

    rag = RAGPipeline()
    rag.build_index(docs, chunk_size=50, persist=False)  # small demo, not saving

    print("Q:", "What is the value of Current Assets?")
    print("A:", rag_qa(rag, "What is the value of Current Assets?", top_k=2))
    print()
    print("Q:", "What is the Total Assets amount?")
    print("A:", rag_qa(rag, "What is the Total Assets amount?", top_k=2))
    print()
    print("Q:", "How much is Equity?")
    print("A:", rag_qa(rag, "How much is Equity?", top_k=2))