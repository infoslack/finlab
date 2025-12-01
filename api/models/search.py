from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 3
    filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
