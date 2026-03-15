"""JusticeMind FastAPI backend — document upload, Q&A, and management endpoints."""

import io
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import ensure_index, get_index, ingest_document, retrieve, build_context, DOC_REGISTRY
from rag.llm import generate_answer

app = FastAPI(title="JusticeMind API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

VALID_DOC_TYPES = {"contract", "legislation", "judgement"}


@app.on_event("startup")
async def startup() -> None:
    """Initialise Endee index on startup."""
    ensure_index()


class AskRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    doc_type: Optional[str] = None
    top_k: int = 5


@app.get("/health")
def health():
    """Return Endee index stats."""
    try:
        return {"status": "ok", "index": get_index().describe()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/upload")
async def upload(file: UploadFile = File(...), doc_type: str = Form(...)):
    """Accept a PDF or TXT file, ingest into Endee, return chunk count."""
    if doc_type not in VALID_DOC_TYPES:
        raise HTTPException(400, f"doc_type must be one of {VALID_DOC_TYPES}")

    data = await file.read()
    filename = file.filename or "unknown"

    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
        except Exception as e:
            raise HTTPException(422, f"PDF parse error: {e}")
    else:
        try:
            text = data.decode("utf-8")
        except Exception as e:
            raise HTTPException(422, f"Text decode error: {e}")

    if not text.strip():
        raise HTTPException(422, "No text could be extracted")

    doc_id = str(uuid.uuid4())
    try:
        chunk_count = ingest_document(doc_id, filename, doc_type, text)
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"doc_id": doc_id, "filename": filename, "doc_type": doc_type,
            "chunk_count": chunk_count, "message": "Document ingested successfully"}


@app.post("/ask")
def ask(req: AskRequest):
    """Retrieve relevant chunks and generate a cited answer."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    try:
        chunks = retrieve(req.question, req.top_k, req.doc_id, req.doc_type)
    except Exception as e:
        raise HTTPException(500, str(e))

    if not chunks:
        raise HTTPException(404, "No relevant passages found")

    try:
        answer = generate_answer(req.question, build_context(chunks))
    except Exception as e:
        raise HTTPException(500, f"LLM error: {e}")

    return {
        "answer": answer,
        "sources": [{"filename": c["filename"], "doc_type": c["doc_type"], "chunk_idx": c["chunk_idx"],
                     "similarity": round(c["similarity"], 4),
                     "excerpt": c["text"][:250] + "..." if len(c["text"]) > 250 else c["text"]}
                    for c in chunks],
    }


@app.get("/documents")
def list_documents():
    """Return all ingested documents from the registry."""
    return {"documents": list(DOC_REGISTRY.values())}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete all vectors for a document from Endee and remove from registry."""
    if doc_id not in DOC_REGISTRY:
        raise HTTPException(404, "Document not found")
    index = get_index()
    for cid in DOC_REGISTRY[doc_id]["chunk_ids"]:
        try:
            index.delete_vector(cid)
        except Exception:
            pass
    del DOC_REGISTRY[doc_id]
    return {"message": "Document deleted successfully"}
