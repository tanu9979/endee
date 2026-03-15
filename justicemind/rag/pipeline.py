"""RAG pipeline: Endee hybrid index, BM25Encoder, document ingestion, and retrieval."""

import os
import re
import uuid
from typing import Optional

from dotenv import load_dotenv
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Endee()
_index = None
_model = SentenceTransformer("all-MiniLM-L6-v2")
DOC_REGISTRY: dict[str, dict] = {}

CHUNK_WORDS = 350
CHUNK_STEP = 280
MIN_CHUNK_WORDS = 50
BATCH_SIZE = 1000
SPARSE_DIM = 30000


class BM25Encoder:
    """Sparse BM25-style TF encoder mapping tokens to fixed vocabulary indices."""

    def __init__(self) -> None:
        """Initialise with empty vocabulary and unfitted state."""
        self._vocab: dict[str, int] = {}
        self._fitted: bool = False

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from corpus, capping indices at 29999."""
        tokens_all: set[str] = set()
        for text in texts:
            tokens_all.update(re.split(r"[\s\W]+", text.lower()))
        tokens_all.discard("")
        for i, token in enumerate(sorted(tokens_all)):
            self._vocab[token] = min(i, 29999)
        self._fitted = True

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Return (sparse_indices, sparse_values) TF representation for text."""
        if not self._fitted:
            return [], []
        tokens = [t for t in re.split(r"[\s\W]+", text.lower()) if t]
        tf: dict[str, int] = {}
        for t in tokens:
            if t in self._vocab:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            return [], []
        max_tf = max(tf.values())
        return [self._vocab[t] for t in tf], [c / max_tf for c in tf.values()]


bm25_encoder = BM25Encoder()


def ensure_index() -> None:
    """Create the justicemind Endee index if it does not exist."""
    global _index
    try:
        existing = [idx.name for idx in client.list_indexes()]
    except Exception as exc:
        raise RuntimeError(f"Cannot reach Endee server: {exc}") from exc

    if "justicemind" not in existing:
        try:
            client.create_index(
                name="justicemind",
                dimension=384,
                sparse_dim=SPARSE_DIM,
                space_type="cosine",
                precision=Precision.INT8,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create index: {exc}") from exc

    try:
        _index = client.get_index(name="justicemind")
    except Exception as exc:
        raise RuntimeError(f"Failed to get index: {exc}") from exc


def get_index():
    """Return the module-level index reference, initialising if needed."""
    global _index
    if _index is None:
        ensure_index()
    return _index


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start: start + CHUNK_WORDS]
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(" ".join(chunk_words))
        if start + CHUNK_WORDS >= len(words):
            break
        start += CHUNK_STEP
    return chunks


def ingest_document(doc_id: str, filename: str, doc_type: str, text: str) -> int:
    """Chunk, embed, and upsert a document into Endee; register in DOC_REGISTRY."""
    index = get_index()
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    if not bm25_encoder._fitted:
        bm25_encoder.fit(chunks)

    dense_vecs = _model.encode(chunks, normalize_embeddings=True, show_progress_bar=False).tolist()

    records: list[dict] = []
    chunk_ids: list[str] = []

    for i, (chunk, dvec) in enumerate(zip(chunks, dense_vecs)):
        cid = str(uuid.uuid4())
        chunk_ids.append(cid)
        s_idx, s_val = bm25_encoder.encode(chunk)
        records.append({
            "id": cid,
            "vector": dvec,
            "sparse_indices": s_idx,
            "sparse_values": s_val,
            "meta": {"doc_id": doc_id, "filename": filename, "doc_type": doc_type, "chunk_idx": i, "text": chunk},
            "filter": {"doc_id": doc_id, "doc_type": doc_type},
        })

    try:
        for start in range(0, len(records), BATCH_SIZE):
            index.upsert(records[start: start + BATCH_SIZE])
    except Exception as exc:
        raise RuntimeError(f"Endee upsert failed: {exc}") from exc

    DOC_REGISTRY[doc_id] = {
        "doc_id": doc_id, "filename": filename, "doc_type": doc_type,
        "chunk_count": len(chunks), "chunk_ids": chunk_ids, "size_bytes": len(text.encode()),
    }
    return len(chunks)


def retrieve(question: str, top_k: int = 5, doc_id: Optional[str] = None, doc_type: Optional[str] = None) -> list[dict]:
    """Embed question and run hybrid Endee query; return top-K chunk dicts."""
    index = get_index()
    dense_vec = _model.encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
    q_idx, q_val = bm25_encoder.encode(question)

    kwargs: dict = {"vector": dense_vec, "sparse_indices": q_idx, "sparse_values": q_val, "top_k": top_k}
    if doc_id is not None:
        kwargs["filter"] = [{"doc_id": {"$eq": doc_id}}]
        kwargs["filter_boost_percentage"] = 20
    elif doc_type is not None:
        kwargs["filter"] = [{"doc_type": {"$eq": doc_type}}]
        kwargs["filter_boost_percentage"] = 15

    try:
        results = index.query(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"Endee query failed: {exc}") from exc

    hits: list[dict] = []
    for r in results:
        meta = r.meta if hasattr(r, "meta") else r.get("meta", {})
        sim = r.similarity if hasattr(r, "similarity") else r.get("similarity", 0.0)
        hits.append({
            "text": meta.get("text", ""), "filename": meta.get("filename", ""),
            "doc_type": meta.get("doc_type", ""), "chunk_idx": meta.get("chunk_idx", 0), "similarity": sim,
        })
    return hits


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string for the LLM."""
    parts = [f"[{i}] (source: {c['filename']} | type: {c['doc_type']}, chunk {c['chunk_idx']})\n{c['text']}"
             for i, c in enumerate(chunks, 1)]
    return "\n\n---\n\n".join(parts)
