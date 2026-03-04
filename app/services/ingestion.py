import os
import shutil
from fastapi import UploadFile

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.vector_store import VectorStoreService


class IngestionService:
    def __init__(self):
        self.vector_store = VectorStoreService()

        # Policy-aware chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Large enough for full policy clauses
            chunk_overlap=200,    # Preserve cross-clause references
            separators=["\n\n", "\n", ".", " ", ""],
        )

    async def process_file(self, file: UploadFile, category: str):
        """
        1. Save uploaded file
        2. Load document (PDF / TXT)
        3. Add metadata
        4. Chunk document
        5. Store embeddings in ChromaDB
        """

        # 1️⃣ Save file locally
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2️⃣ Load document
        if file.filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        raw_docs = loader.load()

        # 3️⃣ Add metadata
        for doc in raw_docs:
            doc.metadata["category"] = category
            doc.metadata["source"] = file.filename

        # 4️⃣ Chunking
        chunks = self.text_splitter.split_documents(raw_docs)

        # 5️⃣ Indexing
        self.vector_store.add_documents(chunks)

        return {
            "filename": file.filename,
            "status": "ingested",
            "chunks_created": len(chunks),
        }
