from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.services.ingestion import IngestionService
from app.services.llm_engine import RAGEngine

router = APIRouter()

# -------- Schemas --------
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


# -------- Services (Singleton-style) --------
ingestion_service = IngestionService()
rag_engine = RAGEngine()


# -------- Endpoints --------
@router.post("/ingest/document")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(..., description="Personal | Home | Business"),
):
    """
    Upload a loan policy document (PDF or TXT).
    """
    try:
        return await ingestion_service.process_file(file, category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_with_policy(request: ChatRequest):
    """
    Ask a question about uploaded loan policy documents.
    """
    try:
        answer = rag_engine.get_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
