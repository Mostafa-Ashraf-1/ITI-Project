from fastapi import APIRouter

from app.schemas.query import QueryRequest, QueryResponse, SourceItem, HealthResponse
from app.services import retrieval, generation

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    status = retrieval.get_status()
    return HealthResponse(
        status="ok",
        collection="egyptian_civil_code",
        num_chunks=status["num_chunks"],
        embedding_backend=status["backend"],
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    contexts = retrieval.retrieve(req.question, top_k=req.top_k)
    answer, backend = generation.generate_answer(req.question, contexts)

    sources = [
        SourceItem(
            article=c.get("article") if c.get("article") not in (None, -1) else None,
            page=c.get("page"),
            section=c.get("section") or None,
            text_preview=c["text"][:200].replace("\n", " "),
        )
        for c in contexts
    ]

    return QueryResponse(
        question=req.question,
        answer=answer,
        sources=sources,
        generation_backend=backend,
    )
