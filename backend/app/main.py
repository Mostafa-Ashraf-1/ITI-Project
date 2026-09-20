from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes.query import router as query_router
from app.services import retrieval
from app.utils.logging_config import configure_logging

logger = configure_logging()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    info = retrieval.init_store()
    logger.info(f"Vector store loaded once at startup: {info}")


app.include_router(query_router)


@app.get("/")
def root():
    return {"message": settings.APP_NAME, "docs": "/docs"}
