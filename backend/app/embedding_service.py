"""
Deterministic, zero-server-key local semantic embedding service for Codebase Intelligence.
Powered by fastembed (BAAI/bge-small-en-v1.5, 384-dimensional dense vectors).
Optimized for memory-constrained environments (Render 512MB RAM free tier).
"""
from __future__ import annotations

import os

# CRITICAL FOR 512MB RAM LIMITS: Prevent ONNX / NumPy from spawning multi-threaded 
# memory pools that trigger Out-Of-Memory (OOM) kills on Render free tier.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import Any, Dict, List, Optional
import structlog
import numpy as np

log = structlog.get_logger()

_EMBED_MODEL = None
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Check if embeddings should be disabled entirely to conserve RAM on free tier
DISABLE_EMBEDDINGS = os.getenv("DISABLE_EMBEDDINGS", "false").lower() == "true"


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Lazy singleton initializer for fastembed local model with low-memory defaults."""
    global _EMBED_MODEL
    if DISABLE_EMBEDDINGS:
        return None
        
    if _EMBED_MODEL is None:
        try:
            from fastembed import TextEmbedding
            log.info("embedding_service.loading_model", model_name=model_name)
            # Initialize with single-threaded execution provider to minimize RAM overhead
            _EMBED_MODEL = TextEmbedding(
                model_name=model_name,
                providers=["CPUExecutionProvider"]
            )
            log.info("embedding_service.model_ready", model_name=model_name)
        except Exception as exc:
            log.error("embedding_service.model_load_failed", error=str(exc))
            raise RuntimeError(
                f"Failed to initialize fastembed model '{model_name}'. "
                "Ensure 'fastembed' is installed: pip install fastembed"
            ) from exc
    return _EMBED_MODEL


def build_symbol_embedding_text(doc: Dict[str, Any]) -> str:
    """
    Constructs a high-signal semantic text representation for a code symbol.
    Combines name, type, file path, signature, docstring, and key dependencies.
    """
    sym_type = doc.get("type", "symbol")
    name = doc.get("name", "")
    file_path = doc.get("file_path", "")
    signature = doc.get("signature", "")
    docstring = (doc.get("docstring") or "").strip()
    calls = doc.get("calls", [])

    parts = [
        f"{sym_type.capitalize()}: {name}",
        f"Path: {file_path}",
    ]
    if signature:
        first_sig_line = signature.split("\n")[0].strip()
        parts.append(f"Signature: {first_sig_line}")
    if docstring:
        truncated_doc = docstring[:200].replace("\n", " ").strip()  # Shortened to save memory strings
        parts.append(f"Description: {truncated_doc}")
    if calls:
        call_names = [c.split(":")[-1] for c in calls[:3]]  # Reduced to top 3
        parts.append(f"Interacts with: {', '.join(call_names)}")

    return " | ".join(parts)


def generate_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """Generates 384-dim dense vectors using a smaller batch size to prevent RAM spikes."""
    if DISABLE_EMBEDDINGS or not texts:
        return []
        
    model = get_embedding_model()
    if not model:
        return []
        
    try:
        # Smaller batch size (default 16 instead of 64) keeps memory footprint flat
        embeddings = list(model.embed(texts, batch_size=batch_size))
        return [e.tolist() for e in embeddings]
    except Exception as exc:
        log.error("embedding_service.generation_failed", error=str(exc))
        return []


def generate_query_embedding(query: str) -> List[float]:
    """Generates embedding for a single search query."""
    embeddings = generate_embeddings([query], batch_size=1)
    if not embeddings:
        return []
    return embeddings[0]


def rank_by_cosine_similarity(
    query_vector: List[float],
    candidate_embeddings: List[List[float]],
    top_k: int = 10,
    min_score: float = 0.0,
) -> List[tuple[int, float]]:
    """
    Calculates cosine similarity between query_vector and an array of candidate embeddings.
    Returns list of (index, similarity_score) sorted by score descending.
    """
    if not candidate_embeddings or not query_vector:
        return []

    q_vec = np.array(query_vector, dtype=np.float32)
    matrix = np.array(candidate_embeddings, dtype=np.float32)

    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        return []

    m_norms = np.linalg.norm(matrix, axis=1)
    m_norms[m_norms == 0] = 1e-9

    similarities = np.dot(matrix, q_vec) / (m_norms * q_norm)

    indices = np.argsort(similarities)[::-1]
    results = []
    for idx in indices:
        score = float(similarities[idx])
        if score < min_score:
            break
        results.append((int(idx), score))
        if len(results) >= top_k:
            break

    return results