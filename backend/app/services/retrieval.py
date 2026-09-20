"""
Retrieval service: loads the persisted vector store ONCE at startup and exposes
retrieve(question, top_k). Supports two backends transparently:

  1. Real ChromaDB collection persisted under settings.VECTOR_STORE_DIR
     (produced when the notebook ran with `chromadb` installed, e.g. in Colab).
  2. The local fallback store (a `<collection>_embeddings.npy` + `<collection>_store.json`
     pair) produced when chromadb/sentence-transformers were not installable.

The active backend is auto-detected at import time so the API code never needs to
change between environments.
"""
import json
import os
import pickle

import numpy as np

from app.core.config import settings

_backend = None
_collection = None
_embed_query = None
_store = None
_mat_norm = None


def _load_query_embedder():
    """Return a function question -> np.ndarray using whichever embedding backend
    produced the persisted vectors (sentence-transformers, else TF-IDF+SVD fallback)."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.EMBEDDING_MODEL)

        def embed(q: str):
            return model.encode([q], normalize_embeddings=True)[0]

        return embed, "sentence-transformers"
    except ImportError:
        vec_path = os.path.join(settings.DATA_DIR, "vectorizer.pkl")
        with open(vec_path, "rb") as f:
            bundle = pickle.load(f)
        vectorizer, svd = bundle["vectorizer"], bundle["svd"]
        from sklearn.preprocessing import normalize

        def embed(q: str):
            tfidf = vectorizer.transform([q])
            reduced = svd.transform(tfidf)
            return normalize(reduced)[0]

        return embed, "sklearn-tfidf-svd (fallback)"


def init_store():
    """Load the vector store once. Call this at FastAPI startup."""
    global _backend, _collection, _embed_query, _store, _mat_norm

    _embed_query, embed_backend = _load_query_embedder()

    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)
        _collection = client.get_collection(settings.COLLECTION_NAME)
        _backend = "chromadb"
    except Exception:
        store_path = os.path.join(settings.VECTOR_STORE_DIR, f"{settings.COLLECTION_NAME}_store.json")
        emb_path = os.path.join(settings.VECTOR_STORE_DIR, f"{settings.COLLECTION_NAME}_embeddings.npy")
        with open(store_path, encoding="utf-8") as f:
            _store = json.load(f)
        mat = np.load(emb_path)
        _mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10)
        _backend = "local-fallback"

    return {"backend": _backend, "embed_backend": embed_backend}


def get_status():
    n = _collection.count() if _backend == "chromadb" else (len(_store["ids"]) if _store else 0)
    return {"backend": _backend, "num_chunks": n}


def retrieve(question: str, top_k: int = None):
    if _embed_query is None:
        init_store()
    top_k = top_k or settings.DEFAULT_TOP_K
    qemb = _embed_query(question)

    if _backend == "chromadb":
        res = _collection.query(query_embeddings=[qemb.tolist()], n_results=top_k)
        results = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({
                "text": doc,
                "article": meta.get("article"),
                "page": meta.get("page"),
                "section": meta.get("section"),
                "distance": dist,
            })
        return results

    q = qemb / (np.linalg.norm(qemb) + 1e-10)
    sims = _mat_norm @ q
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in top_idx:
        meta = _store["metadatas"][i]
        results.append({
            "text": _store["documents"][i],
            "article": meta.get("article"),
            "page": meta.get("page"),
            "section": meta.get("section"),
            "distance": float(1 - sims[i]),
        })
    return results
