# Prompt template for extracting financial information
EXTRACTION_PROMPT = """
You are an assistant that extracts financial details from company balance sheets.
If asked about a financial field (like Total Assets, Current Assets, Equity), 
you should provide the value exactly as present in the document snippet.

Question: {question}
Document snippet: {context}

Answer:
"""

# Prompt template for classification
CLASSIFICATION_PROMPT = """
You are an AI that classifies documents into categories such as Balance Sheet, 
Policy Document, Claim Form, etc. Use only the text provided.

Document snippet: {context}

Category:
"""