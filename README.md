# Employee / Loan Policy RAG System

A production-ready Retrieval-Augmented Generation (RAG) system designed to query and extract insights from policy documents using semantic search and open-source LLMs.

---

## Overview

This project enables users to ask natural language questions about policy documents (e.g., loan policies, employee policies) and receive accurate, context-grounded answers.

Instead of relying on static keyword search, the system uses:

- Semantic embeddings  
- Vector search (ChromaDB)  
- LLM reasoning (Llama 3)  

to retrieve relevant clauses and generate precise answers.

---

## Key Features

- Semantic search over documents  
- LLM-powered answer generation  
- PDF/Text document ingestion  
- Context-aware chunking (1000 tokens, 200 overlap)  
- FastAPI-based scalable backend  
- Metadata tagging for filtering (category, source)  
- Modular architecture (ingestion, retrieval, generation)  

---

## Architecture
User Query → Retriever (ChromaDB) → Context → LLM (Llama 3) → Answer

### Core Components

- **Embedding Model:** BAAI/bge-small-en-v1.5  
- **Vector Database:** ChromaDB  
- **LLM:** Llama 3 (via Ollama)  
- **Backend:** FastAPI  
- **Framework:** LangChain  

---

## Design Decisions

### Chunking Strategy (1000 + 200 overlap)
Ensures policy clauses remain intact and context is preserved.

### Low Temperature (0.1)
Improves factual accuracy for compliance/policy use cases.

### Open-source Stack
Fully reproducible without paid APIs.

### Modular Services Architecture
Easy to scale and extend.

---

## Use Cases

- Employee policy Q&A  
- Loan/financial policy analysis  
- Legal document search  
- Internal knowledge assistants  

---

## Tech Stack

- Python, FastAPI  
- LangChain  
- ChromaDB  
- HuggingFace Embeddings  
- Ollama (Llama 3)  
- Pydantic  

---

## Future Improvements

- Hybrid search (BM25 + vector)  
- Reranking models  
- UI (React / Streamlit)  
- Multi-document comparison  
- Cloud deployment  
