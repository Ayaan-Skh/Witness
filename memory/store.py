"""
ChromaDB Vector Memory

WHAT THIS DOES
------------------------
Stores historical InvestigationBrief summaries as vector embeddings in
ChromaDB. When the agent processes a new cluster, it queries this store
for the 3 most similar historical cases — giving the LLM context like:

  "The same region had a VEGETATION_LOSS + TONE_CRASH combination in
   November 2020 which was later confirmed as the start of the Tigray
   conflict. Confidence was HIGH. The brief was reviewed and published."

This prevents the LLM from treating every alert as unprecedented. An
experienced analyst consults their case files before writing a new memo.
This is that case file system.

EMBEDDING STRATEGY
------------------------
We embed a structured text representation of each cluster:
  "Region: eth_tigray | Sources: SATELLITE, GDELT | Signals: VEGETATION_LOSS,
   TONE_CRASH | Confidence: HIGH | Time: 2020-11 | Summary: ..."

This structured format ensures the embedding captures:
  - Geographic identity (region_id)
  - Which sources fired
  - What kind of anomaly
  - The outcome (confidence, whether it became a real event)

We use sentence-transformers (local, free) rather than OpenAI embeddings
to avoid API costs for every retrieval. The model 'all-MiniLM-L6-v2' is
80MB, fast, and accurate enough for this retrieval task.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import CHROMA_PERSIST_DIR, MEMORY_RETRIEVAL_TOP_K
from normalization.schema import InvestigationBrief, AnomalyEvent

log = logging.getLogger("witness.memory")

# ── Optional dependencies ────────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None
    Settings = None
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    ST_AVAILABLE = False

# Embedding model — loaded once at module level, reused across calls
_embedding_model = None
_chroma_client   = None
_collection      = None

COLLECTION_NAME  = "investigation_briefs"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 80MB, fast, free


# ─────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────

def _get_embedding_model():
    """Lazy-load the sentence transformer. Only loads once per process."""
    global _embedding_model
    if _embedding_model is None:
        if not ST_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    """Lazy-initialise ChromaDB client and collection."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    if not CHROMA_AVAILABLE:
        raise RuntimeError(
            "chromadb not installed. Run: pip install chromadb"
        )

    persist_dir = str(Path(CHROMA_PERSIST_DIR).resolve())
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    # get_or_create — safe to call on every startup
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
    )

    log.info(
        f"ChromaDB collection '{COLLECTION_NAME}' ready. "
        f"Documents: {_collection.count()}"
    )
    return _collection


# ─────────────────────────────────────────────
# TEXT REPRESENTATION
# ─────────────────────────────────────────────

def _cluster_to_text(
    cluster: dict,
    events: Optional[list[AnomalyEvent]] = None,
) -> str:
    """
    Converts a cluster dict into a structured text string for embedding.

    The format is deliberately semi-structured rather than free prose so
    that the embedding captures key fields reliably. Prose summaries can
    bury important identifiers; structured text surfaces them.
    """
    sources  = sorted(set(cluster.get("sources", [])))
    signals  = []
    if events:
        signals = sorted(set(e.signal_type.value for e in events))

    t_start = cluster.get("time_start")
    t_end   = cluster.get("time_end")

    time_str = ""
    if t_start:
        ts = t_start if isinstance(t_start, datetime) else datetime.fromisoformat(str(t_start))
        time_str = ts.strftime("%Y-%m")

    lines = [
        f"Region: {cluster.get('region_id', 'unknown')}",
        f"Sources: {', '.join(sources) if sources else 'unknown'}",
        f"Signals: {', '.join(signals) if signals else 'unknown'}",
        f"Time: {time_str}",
        f"Event count: {len(events) if events else len(cluster.get('events', []))}",
    ]
    return " | ".join(lines)


def _brief_to_text(brief: InvestigationBrief) -> str:
    """
    Converts an InvestigationBrief into a structured text string for embedding.
    Stored after brief generation so future queries can retrieve it.
    """
    streams  = sorted(set(brief.contributing_streams))
    time_str = brief.time_window_start.strftime("%Y-%m")

    lines = [
        f"Region: {brief.region_id}",
        f"Sources: {', '.join(streams)}",
        f"Confidence: {brief.confidence_tier.value}",
        f"Score: {brief.confidence_score:.2f}",
        f"Time: {time_str}",
        f"Status: {brief.status.value}",
    ]

    # Include the first 300 chars of agent reasoning to capture semantic content
    if brief.agent_reasoning:
        reasoning_snippet = brief.agent_reasoning[:300].replace("\n", " ")
        lines.append(f"Summary: {reasoning_snippet}")

    return " | ".join(lines)


def _embed(text: str) -> list[float]:
    """Embeds a single text string. Returns a list of floats."""
    model  = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ─────────────────────────────────────────────
# STORE OPERATIONS
# ─────────────────────────────────────────────

def store_brief(brief: InvestigationBrief) -> str:
    """
    Embeds and stores an InvestigationBrief in ChromaDB.

    Called after a brief is generated and saved to PostgreSQL.
    Returns the document ID used in ChromaDB.

    Metadata stored alongside the embedding (for filtering):
      region_id, confidence_tier, status, time_window_start, stream_count
    """
    collection = _get_collection()
    text       = _brief_to_text(brief)
    embedding  = _embed(text)
    doc_id     = f"brief_{brief.brief_id}"

    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "brief_id":        brief.brief_id,
            "region_id":       brief.region_id,
            "confidence_tier": brief.confidence_tier.value,
            "confidence_score": brief.confidence_score,
            "status":          brief.status.value,
            "time_window":     brief.time_window_start.strftime("%Y-%m"),
            "stream_count":    brief.stream_count,
            "created_at":      brief.created_at.isoformat(),
        }],
    )
    log.debug(f"Stored brief {brief.brief_id} in ChromaDB")
    return doc_id


def retrieve_similar_cases(
    cluster: dict,
    events: Optional[list[AnomalyEvent]] = None,
    top_k: int = MEMORY_RETRIEVAL_TOP_K,
    region_id: Optional[str] = None,
) -> list[dict]:
    """
    Retrieves the top-k most similar historical cases for a cluster.

    Similarity is cosine distance between the cluster's embedding and
    all stored brief embeddings.

    Args:
        cluster:   The cluster dict from cluster_anomalies node.
        events:    AnomalyEvents in the cluster (for richer embedding).
        top_k:     How many historical cases to return.
        region_id: If provided, bias toward same-region cases by retrieving
                   top_k*3 and then sorting by region match + similarity.

    Returns:
        List of dicts, each containing:
          { 'document': str, 'metadata': dict, 'distance': float }
        Sorted by similarity (most similar first).
        Empty list if ChromaDB has fewer than 3 documents.
    """
    collection = _get_collection()

    if collection.count() < 3:
        log.debug("ChromaDB has fewer than 3 documents — skipping retrieval")
        return []

    text      = _cluster_to_text(cluster, events)
    embedding = _embed(text)

    # Retrieve more candidates if we want to bias by region
    n_retrieve = top_k * 3 if region_id else top_k

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_retrieve, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    cases = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        cases.append({
            "document": doc,
            "metadata": meta,
            "distance": dist,          # cosine distance: 0 = identical, 2 = opposite
            "similarity": 1 - dist,   # higher = more similar
        })

    # Sort: same-region cases first, then by similarity
    if region_id:
        cases.sort(
            key=lambda c: (
                c["metadata"].get("region_id") != region_id,  # False (0) = same region first
                c["distance"],
            )
        )
        cases = cases[:top_k]

    return cases


def format_historical_context(cases: list[dict]) -> str:
    """
    Formats retrieved historical cases into a natural language summary
    for inclusion in the Gemini prompt.

    Returns a string like:
      "Similar historical cases:
       1. [2020-11] eth_tigray — HIGH confidence, SATELLITE+GDELT,
          score=0.82, status=PUBLISHED. [summary snippet]
       2. ..."

    Returns "No relevant historical cases found." if cases is empty.
    """
    if not cases:
        return "No relevant historical cases found in memory."

    lines = ["Similar historical cases from memory:"]
    for i, case in enumerate(cases, 1):
        meta    = case.get("metadata", {})
        doc     = case.get("document", "")
        sim     = case.get("similarity", 0)

        region    = meta.get("region_id",       "unknown")
        time_win  = meta.get("time_window",     "unknown")
        tier      = meta.get("confidence_tier", "unknown")
        score     = meta.get("confidence_score", 0)
        status    = meta.get("status",          "unknown")
        streams   = meta.get("stream_count",    0)

        # Extract summary snippet from document text
        summary_part = ""
        if "Summary:" in doc:
            summary_part = doc.split("Summary:")[-1].strip()[:150]

        line = (
            f"{i}. [{time_win}] {region} — {tier} confidence "
            f"(score={score:.2f}), {streams} sources, status={status}"
        )
        if summary_part:
            line += f"\n   Context: {summary_part}"

        lines.append(line)

    return "\n".join(lines)


def get_store_stats() -> dict:
    """Returns statistics about the ChromaDB collection."""
    if not CHROMA_AVAILABLE:
        return {"available": False}
    try:
        collection = _get_collection()
        return {
            "available":    True,
            "document_count": collection.count(),
            "collection":   COLLECTION_NAME,
            "persist_dir":  CHROMA_PERSIST_DIR,
            "model":        EMBEDDING_MODEL,
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


def clear_store() -> int:
    """Deletes all documents from the collection. Used for testing."""
    collection = _get_collection()
    count = collection.count()
    if count > 0:
        all_ids = collection.get()["ids"]
        collection.delete(ids=all_ids)
    return count