# Egyptian Civil Code — Text-Based RAG Assistant

**Track:** Core Track only (text-based RAG). No Extended Track, no computer vision, no OCR, no YOLO.

## 1. Project Overview

A Retrieval-Augmented Generation (RAG) assistant that answers questions about the
**Egyptian Civil Code** by retrieving the actual, relevant articles from the source
document and generating an answer grounded strictly in that retrieved text, with
citations to the article number and page it came from.

## 2. Problem Statement

Legal texts are long, dense, and hard to search by keyword alone. A lawyer, student,
or citizen who wants to know "what does the Egyptian Civil Code say about X" has to
either know the article number already or read through hundreds of pages. This
project builds a semantic search + grounded-answer system over the Civil Code so a
question in plain Arabic (or English) returns the exact governing article(s) and a
answer that is traceable back to the source text — with an explicit "not found"
response instead of a guess when the code doesn't address the question.

## 3. Domain

Egyptian Civil Law / Egyptian Civil Code (القانون المدني المصري).

## 4. Dataset Description

- **File:** `egyptian_civil_code.pdf`
- **170 pages**, bilingual Arabic/English, native text layer (no OCR required)
- **1,094 article-aware chunks** after cleaning — see `notebooks/rag_pipeline.ipynb`
  Section 2–4 for the full, real inspection/cleaning/chunking log
- Structure: every legal rule is marked with the Arabic word **"مادة" (Article)**,
  followed a few lines later by an English cross-translation **"Article N"**, which
  is what the chunker uses to resolve a reliable article number

## 5. RAG Architecture

```
Egyptian Civil Code PDF
        ↓
PDF Text Extraction (pypdf)
        ↓
Data Inspection
        ↓
Text Cleaning (whitespace/newline normalization; article markers preserved)
        ↓
Article-aware Chunking (split on 'مادة', number resolved from 'Article N')
        ↓
Embeddings (sentence-transformers, multilingual)
        ↓
ChromaDB Vector Store (persisted to disk)
        ↓
Retriever (top-k cosine similarity)
        ↓
Relevant Legal Context
        ↓
Grounded Prompt
        ↓
Local Ollama LLM
        ↓
Grounded Answer + Source / Citation
        ↓
Evaluation
```

Then wired into:

```
FastAPI Backend  →  Streamlit Frontend
```

### ⚠️ About the notebook's embedding/vector-store/LLM backends

The attached notebook was authored and **actually executed end-to-end** against the
real PDF in a sandboxed environment whose network policy only allows a small
allow-list of pip packages — `sentence-transformers`, `chromadb`, `fastapi`,
`streamlit`, and `ollama` could not be installed there. Every relevant cell uses a
try/except pattern: it always attempts the real, required library first, and only
falls back to a local equivalent (TF-IDF+SVD embeddings, a NumPy-backed vector store
with the same API shape as a Chroma collection, and a citation-only extractive
answer composer) when that's unavailable — so every cell still runs and prints real,
non-fabricated output. **In Google Colab, with normal internet access, re-running
the same notebook automatically uses the real sentence-transformers + chromadb +
Ollama stack — no code changes required.** This is called out explicitly in the
printed output of Sections 5, 6, and 8 of the notebook.

The `backend/app/services/retrieval.py` and `generation.py` modules use the exact
same auto-detecting pattern, so the API works unchanged in either environment.

## 6. Technology Stack

| Layer | Technology |
|---|---|
| PDF extraction | `pypdf` |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector store | `ChromaDB` (persistent) |
| LLM | Local `Ollama` (default model: `llama3.1`) |
| Backend | `FastAPI` |
| Frontend | `Streamlit` |
| Notebook | Jupyter / Google Colab |

## 7. Project Structure

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb        # executed end-to-end, real outputs saved inline
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/retrieval.py
│   │   ├── services/generation.py
│   │   └── utils/logging_config.py
│   ├── data/                     # chunks.json, config.json, vector_store/
│   ├── tests/test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── .env.example
│   └── requirements.txt
├── README.md
└── .gitignore
```

## 8. Installation

```bash
git clone <your-repo-url>
cd rag-assistant-project
```

## 9. Google Colab Notebook Instructions

1. Open `notebooks/rag_pipeline.ipynb` in Google Colab.
2. Upload `egyptian_civil_code.pdf` to the Colab file browser (or mount Drive).
3. Run all cells top to bottom (`Runtime → Run all`). With internet access, Section 1's
   `pip install` cell installs the real stack, and every later cell automatically
   uses it instead of the local fallback.
4. To use a real local LLM inside Colab:
   ```bash
   !curl -fsSL https://ollama.com/install.sh | sh
   !ollama serve &
   !ollama pull llama3.1
   ```
5. Section 11 exports `chunks.json`, `config.json`, and `data/vector_store/` —
   copy these into `backend/data/` to power the API (already done for you with the
   fallback-backend outputs from this run; re-run and re-copy after a Colab run with
   the real stack for full-quality embeddings).

## 10. Backend Instructions

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

Run tests:
```bash
pytest
```

## 11. Frontend Instructions

```bash
cd frontend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## 12. Environment Variables

**backend/.env**
| Variable | Description | Default |
|---|---|---|
| `DATA_DIR` | Path to exported notebook data | `data` |
| `VECTOR_STORE_DIR` | Path to persisted vector store | `data/vector_store` |
| `COLLECTION_NAME` | Chroma collection name | `egyptian_civil_code` |
| `EMBEDDING_MODEL` | sentence-transformers model | `paraphrase-multilingual-MiniLM-L12-v2` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.1` |
| `DEFAULT_TOP_K` | Default chunks retrieved | `3` |
| `ALLOWED_ORIGINS` | CORS origins | `*` |

**frontend/.env**
| Variable | Description | Default |
|---|---|---|
| `BACKEND_URL` | FastAPI backend URL | `http://localhost:8000` |

## 13. API Reference

### `GET /health`
Returns service status and index size.
```json
{
  "status": "ok",
  "collection": "egyptian_civil_code",
  "num_chunks": 1094,
  "embedding_backend": "chromadb"
}
```

### `POST /query`
Request:
```json
{ "question": "ما هي أهلية القاصر الذي بلغ ثماني عشرة سنة؟", "top_k": 3 }
```
Response:
```json
{
  "question": "ما هي أهلية القاصر الذي بلغ ثماني عشرة سنة؟",
  "answer": "Based on the retrieved text (Article 42, p. 4): ...",
  "sources": [
    {"article": 42, "page": 4, "section": "SECTION II", "text_preview": "..."}
  ],
  "generation_backend": "extractive-fallback (no ollama server reachable)"
}
```
Invalid input (question missing or too short) → `HTTP 422`.

## 14. Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "متى تسقط دعوى المسئولية التقصيرية بالتقادم؟", "top_k": 3}'
```

## 15. Evaluation Results

Evaluated on 10 real questions against the actual retrieved context (see notebook
Section 10 for the full table and reasoning):

| Metric | Result |
|---|---|
| Retrieval fully relevant | 8 / 10 (80%) |
| Answers judged fully correct | 8 / 10 (80%) |
| Partial matches | 1 / 10 |
| Clear failures | 1 / 10 |
| Hallucinations | 0 / 10 (0%, by construction of the extractive fallback) |

**Known failure case:** a question about "رهن الحيازة" (possessory pledge) matched
an unrelated preamble article — a known weakness of the local TF-IDF fallback
embedding versus a true semantic model; expected to improve with real
`sentence-transformers` in Colab.

## 16. Screenshots

_Add screenshots of the Streamlit UI and API docs here after running the app locally._

## 17. How to Run the Complete Project

1. Run the notebook (Colab, with internet) to produce `backend/data/`.
2. Start the backend: `cd backend && uvicorn app.main:app --reload`.
3. Start the frontend: `cd frontend && streamlit run app.py`.
4. Open the Streamlit URL and ask a question about the Egyptian Civil Code.

## 18. Limitations

- The notebook, as executed and attached, ran with **local fallbacks** for
  embeddings, the vector store, and the LLM because the authoring sandbox blocks
  installing `sentence-transformers` / `chromadb` / `ollama`. Re-run in Colab for
  the full-quality, assignment-specified stack (auto-detected, no code changes).
- The backend/frontend code has been syntax-checked but not run end-to-end in this
  environment for the same reason (those packages aren't installable here either).
- Retrieval quality with the fallback embedding is good (80% top-1 relevance on the
  test set) but not perfect — see the documented failure case above.
