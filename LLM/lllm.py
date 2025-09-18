import os
from llm.prompts import FINANCIAL_QA_PROMPT

# (Placeholder: replace with actual API call if needed, e.g. OpenAI, HuggingFace, etc.)
def query_llm(context: str, question: str) -> str:
    """
    Query an LLM with the given context and question.
    For now, this is a placeholder. Replace with your preferred LLM API.
    """
    prompt = FINANCIAL_QA_PROMPT.format(context=context, question=question)

    # Instead of calling API, return a dummy response for now
    return f"[Mock Answer] Based on context: {context[:100]}... \nQuestion: {question}"


def rag_qa(rag_pipeline, question, top_k=3):
    """
    End-to-end RAG-based QA:
    1. Retrieve relevant chunks
    2. Pass to LLM for final answer
    """
    chunks = rag_pipeline.retrieve(question, top_k=top_k)
    context = "\n".join([c["chunk"] for c in chunks])

    return query_llm(context, question)


if __name__ == "__main__":
    # Demo (mock)
    dummy_docs = [{"file": "doc1.html", "text": "Total Assets 10000. Equity 5000. Current Assets 3000"}]

    from llm.rag import RAGPipeline
    rag = RAGPipeline()
    rag.build_index(dummy_docs)

    print(rag_qa(rag, "What is the Total Assets?"))