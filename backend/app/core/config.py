"""
Application configuration, loaded from environment variables (see .env.example).
"""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "Egyptian Civil Code RAG API"
    APP_VERSION: str = "1.0.0"

    DATA_DIR: str = os.getenv(
        "DATA_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
    )
    VECTOR_STORE_DIR: str = os.getenv(
        "VECTOR_STORE_DIR",
        os.path.join(DATA_DIR, "vector_store"),
    )
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "egyptian_civil_code")

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
