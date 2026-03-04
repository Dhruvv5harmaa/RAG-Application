from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from app.core.config import settings


class VectorStoreService:
    """
    Singleton service for Embeddings + ChromaDB
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print(f"[VectorStore] Loading embeddings: {settings.EMBEDDING_MODEL_NAME}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_db = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="loan_policies",
        )

    def add_documents(self, documents):
        return self.vector_db.add_documents(documents)

    def get_retriever(self, k: int = 4):
        return self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
