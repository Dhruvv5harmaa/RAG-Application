
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
