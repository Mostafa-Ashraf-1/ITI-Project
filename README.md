# Egyptian Civil Code — Text-Based RAG Assistant

**Track:** Core Track only (text-based RAG). No Extended Track, no computer vision, no OCR, no YOLO.

## 1. Project Overview

A Retrieval-Augmented Generation (RAG) assistant for answering questions about the **Egyptian Civil Code**.

The system:

- Extracts the actual legal text from the source PDF.
- Cleans and structures the document into **article-aware chunks**.
- Generates multilingual semantic embeddings using **sentence-transformers**.
- Stores and retrieves legal passages using **ChromaDB**.
- Generates grounded answers from the retrieved legal context using **Groq**.
- Includes article/page citations so retrieved evidence is traceable.
- Supports an explicit fallback response when the required information is not present in the retrieved context.
- Provides a **FastAPI backend** and **Streamlit frontend**.

## 2. Problem Statement

Legal texts are long, dense, and difficult to search with keywords alone. A user may ask a question in natural Arabic or English without knowing the relevant article number.

This project provides semantic retrieval over the Egyptian Civil Code and returns the most relevant legal text together with a grounded answer and source information.

The core principle is **grounded generation**: the answer should be based on retrieved Civil Code text rather than unsupported outside knowledge.

## 3. Domain

**Egyptian Civil Law / Egyptian Civil Code (القانون المدني المصري)**

## 4. Dataset

- **Source:** `egyptian_civil_code.pdf`
- **Pages:** 170
- **Language:** Bilingual Arabic / English
- **Text layer:** Native text; OCR is not required.
- **Chunks:** 1,094 article-aware chunks.
- Legal article boundaries are detected using the Arabic **"مادة"** marker, while the paired English **"Article N"** marker is used to resolve the article number.

## 5. RAG Pipeline

```text
Egyptian Civil Code PDF
        ↓
PDF Text Extraction (pypdf)
        ↓
Inspection & Cleaning
        ↓
Article-aware Chunking
        ↓
Multilingual Embeddings
(sentence-transformers)
        ↓
ChromaDB Vector Store
        ↓
Semantic Retrieval (Top-K)
        ↓
Retrieved Legal Context
        ↓
Grounded Prompt
        ↓
LLM Generation
(Groq / openai/gpt-oss-20b)
        ↓
Answer + Article/Page Sources
        ↓
Evaluation
```

The complete pipeline is implemented and executed in:

`notebooks/rag_pipeline.ipynb`

## 6. Current Model / Backend Configuration

The exported notebook artifacts currently record:

| Component | Current configuration |
|---|---|
| PDF extraction | `pypdf` |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Embedding backend | `sentence-transformers` |
| Embedding dimension | 384 |
| Vector store | `ChromaDB` |
| Vector store path | `backend/data/vector_store/` |
| LLM provider | Groq |
| LLM model | `openai/gpt-oss-20b` |
| Fallback generation | Extractive / retrieved-text based |

The notebook exports the resulting pipeline configuration to:

`backend/data/config.json`

It also records Groq availability/model information in:

`backend/data/groq_status.json`

> **Important:** API keys are not stored in the repository. Configure the Groq API key through environment variables. If Groq is unavailable, the pipeline/API uses the documented extractive fallback instead of inventing unsupported legal information.

## 7. Project Structure

```text
ITI-Project/
├── notebooks/
│   ├── rag_pipeline.ipynb
│   └── data/
│       └── availability.json
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes/query.py
│   │   ├── core/config.py
│   │   ├── schemas/query.py
│   │   ├── services/
│   │   │   ├── retrieval.py
│   │   │   └── generation.py
│   │   └── utils/logging_config.py
│   │
│   ├── data/
│   │   ├── chunks.json
│   │   ├── config.json
│   │   ├── evaluation.csv
│   │   ├── embeddings.npy
│   │   ├── embed_meta.json
│   │   ├── groq_status.json
│   │   ├── rag_results.json
│   │   └── vector_store/
│   │
│   ├── tests/
│   │   └── test_query.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── requirements.txt
│   └── .env.example
│
├── README.md
└── .gitignore
```

## 8. Notebook

The notebook is an **executed end-to-end RAG pipeline**, not only a collection of unexecuted code cells.

It covers:

1. Environment setup and dependency availability.
2. PDF inspection and text extraction.
3. Page-level cleaning.
4. Article-aware chunking.
5. Semantic embedding generation.
6. Vector-store construction.
7. Retrieval experiments.
8. Grounded answer generation.
9. Evaluation on real Civil Code questions.
10. Export of backend-ready artifacts.

The notebook also documents the fallback behavior used when optional services are unavailable.

## 9. Evaluation

The current evaluation uses **10 real questions** about the Egyptian Civil Code.

| Metric | Result |
|---|---:|
| Fully relevant retrieval | **8 / 10 (80%)** |
| Fully correct answers | **8 / 10 (80%)** |
| Partial matches | **1 / 10** |
| Clear failures | **1 / 10** |
| Hallucinations in extractive fallback | **0 / 10** |

### Known retrieval limitations

One clear failure involved the question:

> `ما هي أحكام رهن الحيازة في القانون المدني؟`

The local fallback retrieval missed the semantic connection and retrieved an unrelated preamble article. A second test produced a partial match for a broader question about real rights over immovables.

The notebook identifies these cases explicitly and documents the expected improvement from semantic sentence embeddings.

The detailed evaluation is available in:

`backend/data/evaluation.csv`

## 10. Backend

The backend is implemented with **FastAPI**.

### Main endpoints

#### `GET /health`

Returns the service status and number of indexed chunks.

Example:

```json
{
  "status": "ok",
  "collection": "egyptian_civil_code",
  "num_chunks": 1094,
  "embedding_backend": "chromadb"
}
```

#### `POST /query`

Example request:

```json
{
  "question": "ما هي أهلية القاصر الذي بلغ ثماني عشرة سنة؟",
  "top_k": 3
}
```

The response contains:

- The original question.
- The generated answer.
- Retrieved sources.
- Article number when available.
- Page number.
- Section.
- A short text preview.
- The generation backend used.

Invalid questions are rejected through the Pydantic request schema.

## 11. Backend Installation

```bash
git clone https://github.com/abdooashraf49-arch/ITI-Project.git
cd ITI-Project/backend

python -m venv .venv
```

### Windows

```bash
.venv\\Scripts\\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template:

```bash
cp .env.example .env
```

Run FastAPI:

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://localhost:8000/docs
```

Run tests:

```bash
pytest
```

## 12. Frontend

The frontend is a Streamlit application.

It provides:

- Arabic / English question input.
- Configurable Top-K retrieval.
- Backend connection status.
- Generated answer display.
- Generation backend information.
- Expandable article/page source sections.
- Explicit messaging when the backend cannot be reached.

Run it with:

```bash
cd frontend
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The frontend expects the backend URL through:

`frontend/.env`

with:

```env
BACKEND_URL=http://localhost:8000
```

## 13. Environment Variables

### Backend

| Variable | Description | Default |
|---|---|---|
| `DATA_DIR` | Exported RAG data directory | `data` |
| `VECTOR_STORE_DIR` | Persisted vector-store directory | `data/vector_store` |
| `COLLECTION_NAME` | Chroma collection name | `egyptian_civil_code` |
| `EMBEDDING_MODEL` | Sentence-transformers model | `paraphrase-multilingual-MiniLM-L12-v2` |
| `GROQ_API_KEY` | API key used for Groq LLM generation | — |
| `GROQ_MODEL` | Groq model used for generation | `openai/gpt-oss-20b` |
| `DEFAULT_TOP_K` | Default number of retrieved chunks | `3` |
| `ALLOWED_ORIGINS` | CORS origins | `*` |

### Frontend

| Variable | Description | Default |
|---|---|---|
| `BACKEND_URL` | FastAPI backend URL | `http://localhost:8000` |

## 14. Example API Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"متى تسقط دعوى المسئولية التقصيرية بالتقادم؟","top_k":3}'
```

## 15. Grounding and Safety Design

The project is designed to reduce unsupported legal answers:

- Retrieved context is explicitly included in the generation prompt.
- The generation instructions prohibit inventing article numbers or legal rules.
- Article and page metadata are returned with retrieved passages.
- When the required information is not available in the retrieved context, the system is instructed to say so.
- The extractive fallback only composes its answer from retrieved text.

This project is an educational RAG implementation and **not a substitute for professional legal advice**.

## 16. Data / Artifacts

The repository contains the exported artifacts required by the implemented pipeline, including:

- Article-aware chunks.
- Embedding metadata.
- Embedding arrays.
- Evaluation results.
- Retrieval results.
- Configuration.
- Persisted vector-store files.

Large generated artifacts are kept under `backend/data/` so the backend can load the exported pipeline state directly.

## 17. How to Run the Complete Project

### Option A — Notebook

1. Open `notebooks/rag_pipeline.ipynb`.
2. Use Google Colab or a local Jupyter environment.
3. Provide `egyptian_civil_code.pdf`.
4. Run the notebook from top to bottom.
5. Inspect the retrieval and evaluation outputs.
6. The notebook exports the backend artifacts under `backend/data/`.

### Option B — Full Application

1. Prepare the exported backend data.
2. Install backend dependencies.
3. Start FastAPI.
4. Start Streamlit.
5. Open the Streamlit application.
6. Ask questions in Arabic or English.
7. Inspect the returned answer and source articles/pages.

## 18. Limitations

- Retrieval quality depends on the embedding model and Top-K value.
- The evaluation set contains only 10 questions and should not be treated as a comprehensive benchmark.
- Some broad legal questions may require multiple retrieved articles rather than a single top result.
- Groq is the intended LLM backend for the project, with an extractive fallback when Groq is unavailable.
- This is a technical RAG project over a legal document; it does not provide professional legal advice.

## 19. Future Improvements

- Improve semantic retrieval for difficult legal terminology.
- Add reranking after initial vector retrieval.
- Expand the evaluation set with more legal questions.
- Add conversational multi-turn retrieval.
- Add richer source previews and direct page navigation.
- Unify the notebook's Groq generation path with the deployed API generation service.
- Add authentication and production deployment configuration.

---

**Project:** Egyptian Civil Code Text-Based RAG Assistant  
**Repository:** https://github.com/Mostafa-Ashraf-1/ITI-Project
