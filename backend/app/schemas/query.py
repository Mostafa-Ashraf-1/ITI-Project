from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Legal question about the Egyptian Civil Code")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    article: Optional[int] = None
    page: Optional[int] = None
    section: Optional[str] = None
    text_preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    generation_backend: str


class HealthResponse(BaseModel):
    status: str
    collection: str
    num_chunks: int
    embedding_backend: str
