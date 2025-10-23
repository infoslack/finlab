from fastapi import APIRouter
from models.rag import RAGRequest, RAGResponse
from services.rag import RAGService

from routers.search import search_service

router = APIRouter()
rag_service = RAGService(search_service=search_service)


@router.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest):
    return rag_service.generate_answer(request.query, request.limit)
