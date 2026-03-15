# ⚖️ JusticeMind — AI Legal Document Q&A Engine

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit)
![Endee](https://img.shields.io/badge/Endee-Vector_DB-6C47FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

AI-powered Legal Document Q&A Engine — upload contracts, Indian laws, and court judgements, ask plain-English questions, get precise cited answers in seconds.

---

## Problem Statement

Legal professionals, law students, and researchers waste hours manually searching through contracts, statutes, and court judgements. A single Supreme Court judgement can exceed 200 pages, and keyword search cannot understand the meaning behind a question.

JusticeMind solves this by combining hybrid vector search with a large language model. Upload any legal document, ask a question in plain English, and receive a grounded answer with citations pointing to the exact passages — in seconds, not hours.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION FLOW                           │
│                                                                 │
│  PDF/TXT ──► pypdf extract ──► Chunker (350w / 70w overlap)    │
│                                      │                          │
│                         ┌────────────┴────────────┐            │
│                         ▼                         ▼            │
│                SentenceTransformer           BM25Encoder        │
│                (384-dim dense vec)       (sparse 30k-dim)       │
│                         └────────────┬────────────┘            │
│                                      ▼                          │
│                       Endee hybrid index upsert                 │
│                  (cosine / INT8 / batches of ≤1000)             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          QUERY FLOW                             │
│                                                                 │
│  User ──► Streamlit UI ──► FastAPI /ask                        │
│                                 │                               │
│                     ┌───────────┴───────────┐                  │
│                     ▼                       ▼                  │
│           Dense embed (MiniLM)      BM25 sparse encode          │
│                     └───────────┬───────────┘                  │
│                                 ▼                               │
│                    Endee hybrid query                           │
│             (+ optional doc_id / doc_type filter)               │
│                                 ▼                               │
│                     Top-K chunks + similarity                   │
│                                 ▼                               │
│              Groq llama-3.1-8b-instant                          │
│              (numbered context → cited answer)                  │
│                                 ▼                               │
│          Cited answer + source cards ──► Streamlit UI           │
└─────────────────────────────────────────────────────────────────┘
```

**Ingestion:** Text is extracted, split into 350-word overlapping chunks, encoded into 384-dim dense vectors and 30k-dim sparse BM25 vectors, then upserted into Endee with metadata and filters.

**Query:** The question is encoded with the same models. Endee runs a hybrid query returning top-K chunks. Optional filters scope results to a specific document or type. Chunks are passed to Groq LLM which generates a cited answer.

---

## How Endee Is Used

Endee is the sole vector database. The official Python SDK is used exclusively for all storage and retrieval operations.

| Endee Feature | SDK Method | How JusticeMind Uses It |
|---|---|---|
| Hybrid index creation | `client.create_index(sparse_dim=30000)` | Creates `justicemind` index with 384-dim dense + 30k-dim sparse |
| Batch upsert | `index.upsert([...])` | Ingests chunks with dense + sparse vectors + metadata |
| Hybrid query | `index.query(vector, sparse_indices, sparse_values)` | Retrieves top-K chunks by semantic + keyword relevance |
| Filtered query by doc | `filter=[{"doc_id": {"$eq": id}}]` | Scopes Q&A to a single document |
| Filtered query by type | `filter=[{"doc_type": {"$eq": type}}]` | Scopes Q&A to contracts, legislation, or judgements |
| Delete vector | `index.delete_vector(id)` | Removes chunk vectors when a document is deleted |
| Describe index | `index.describe()` | Powers the `/health` endpoint |
| List indexes | `client.list_indexes()` | Checks if `justicemind` index exists on startup |

---

## Supported Document Types

| Type | Examples | doc_type value |
|---|---|---|
| Contract / NDA | Employment agreements, NDAs, service contracts | `contract` |
| Indian Legislation | IPC, Constitution of India, IT Act, GST Act | `legislation` |
| Court Judgement | Supreme Court, High Court rulings and orders | `judgement` |

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Vector database | Endee (hybrid dense + sparse) | latest |
| Dense embeddings | sentence-transformers `all-MiniLM-L6-v2` | 3.0.1 |
| Sparse embeddings | Custom BM25Encoder (30k-dim) | — |
| LLM | Groq `llama-3.1-8b-instant` | 0.9.0 |
| Backend API | FastAPI + uvicorn | 0.111.0 |
| Frontend | Streamlit | ≥1.40.0 |
| PDF extraction | pypdf | 4.3.1 |
| Orchestration | Docker Compose | — |

---

## Setup Instructions

### Option A: Docker Compose (Recommended)

```bash
git clone https://github.com/YOUR-USERNAME/endee && cd endee/justicemind
cp .env.example .env          # add your GROQ_API_KEY
docker compose up --build
```

- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs
- Endee server → http://localhost:8080

### Option B: Plain Python

```bash
docker run -d -p 8080:8080 -e NDD_AUTH_TOKEN="" endeeio/endee-server:latest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
uvicorn app:app --reload --port 8000   # terminal 1
streamlit run ui.py                    # terminal 2
```

---

## Usage

1. Open http://localhost:8501
2. Upload a PDF or TXT legal document from the sidebar
3. Select the document type (Contract, Indian Law, Court Judgement)
4. Click **⬆️ Upload & Ingest** and wait for chunk count confirmation
5. Click **🔄 Refresh** to see your document listed
6. Choose search scope in the main panel
7. Type your question and click **⚖️ Ask JusticeMind**
8. Read the cited answer and expand source cards for raw excerpts

---

## Example Q&A

**Contract**
> Q: What are the termination clauses?
> A: Per [1], either party may terminate with 30 days written notice. Section 12.2 states termination takes effect immediately on material breach [2]. *Disclaimer: For informational purposes only. Not legal advice.*

**Legislation**
> Q: What is the punishment for theft under IPC?
> A: Section 379 IPC as cited in [1]: "Whoever commits theft shall be punished with imprisonment...which may extend to three years, or with fine, or with both." *Disclaimer: For informational purposes only. Not legal advice.*

**Judgement**
> Q: What was the ratio decidendi?
> A: The court held in [1] that Article 21 extends to the right to a clean environment: "The right to life includes the right to live with human dignity..." [2] *Disclaimer: For informational purposes only. Not legal advice.*

---

## Why RAG Over Agents

RAG was chosen because the problem is closed and well-defined — retrieve relevant passages from a fixed document corpus and answer from them. RAG solves this more reliably, cheaply, and transparently than an agent. No tool use or multi-step planning is needed. Citations are trivial since the source chunks are already in hand, and the system's behaviour is easy to audit.

---

## Why Hybrid Search for Legal Documents

Legal texts contain exact clause numbers, section references like "Section 302 IPC", case citations, and defined terms that pure semantic search misses. BM25 sparse vectors in Endee's hybrid index ensure both exact term matching and semantic similarity work together — so "Section 302" retrieves that exact reference while "punishment for murder" retrieves semantically related passages.

---

## Mandatory Repository Steps

Starred `endee-io/endee` ✅
Forked to `github.com/YOUR-USERNAME/endee` ✅
Project built inside the fork under `/justicemind` ✅

---

## License

MIT License. Copyright (c) 2024 JusticeMind contributors.

**Disclaimer:** JusticeMind is for informational and educational purposes only. It does not constitute legal advice. Always consult a qualified legal professional.
