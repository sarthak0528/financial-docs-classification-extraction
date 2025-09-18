Intelligent Document Classifier & Extractor

Overview
This project demonstrates a complete pipeline for classification and extraction of financial documents.
It combines traditional ML, deep learning, and retrieval-augmented generation (RAG) to process PDFs and HTML financial documents automatically.

Features
📄 Document Classification

Categorizes documents (Balance Sheet, P&L, etc.).

Baseline: TF-IDF + Logistic Regression.

Deep Learning: Embeddings + Dense layers.

🏷 Named Entity Recognition (NER)

Extracts key financial entities like:

Current Assets

Total Assets

Equity

Trained using SpaCy NER with annotated HTML balance sheets.

🤖 Retrieval-Augmented Generation (RAG)

Uses sentence-transformers for embedding chunks.

FAISS for similarity search.

Query examples:

"What is the value of Current Assets?"

"How much is Equity?"
