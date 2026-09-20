"""
Application configuration, loaded from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "Egyptian Civil Code RAG API"
    APP_VERSION: str = "1.0.0"

    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", os.path.join(DATA_DIR, "vector_store"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "egyptian_civil_code")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # Retrieval
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))

    # CORS
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
