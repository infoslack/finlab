from typing import List
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    limit: int = 3


class SearchResult(BaseModel):
    score: float
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
