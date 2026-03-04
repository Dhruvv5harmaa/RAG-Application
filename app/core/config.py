import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Project Info
    PROJECT_NAME: str = "Loan Policy RAG Agent"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Paths
    BASE_DIR: str = os.getcwd()
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "data")
    CHROMA_PERSIST_DIR: str = os.path.join(BASE_DIR, "chroma_db")

    # Embedding Model (Lightweight, CPU-friendly)
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

    # LLM Config (Low-RAM safe model)
    LLM_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL_NAME: str = "tinyllama:1.1b-chat"

    class Config:
        case_sensitive = True


# Initialize settings
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
