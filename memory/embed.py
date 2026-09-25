# memory/embed.py - Text embedding via fastembed (ONNX, CPU)
#
# Wraps fastembed's TextEmbedding with a lazy singleton. The model is
# ~90MB and downloaded once on first use. If anything fails, embed()
# returns None and the caller falls back to non-semantic retrieval.

import threading
from typing import Optional

MODEL_ID = "BAAI/bge-small-en-v1.5"
DIM = 384

_model = None
_model_lock = threading.Lock()
_model_failed = False


def _load_model():
    global _model, _model_failed
    if _model is not None:
        return _model
    if _model_failed:
        return None
    with _model_lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            print(f"🧠 Loading embedding model: {MODEL_ID}")
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL_ID)
            print("✅ Embedding model ready.")
            return _model
        except Exception as e:
            print(f"⚠️ Embedding model unavailable: {e}")
            print("   Falling back to recency-based retrieval.")
            _model_failed = True
            return None


def embed(text: str) -> Optional[list[float]]:
    """Return a 384-dim embedding, or None if the model is unavailable."""
    model = _load_model()
    if model is None or not text:
        return None
    try:
        return next(model.embed([text])).tolist()
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return None


def embed_batch(texts: list[str]) -> Optional[list[list[float]]]:
    """Batch embed. Returns None if the model is unavailable."""
    model = _load_model()
    if model is None or not texts:
        return None
    try:
        return [e.tolist() for e in model.embed(texts)]
    except Exception as e:
        print(f"⚠️ Batch embed failed: {e}")
        return None


def is_available() -> bool:
    return _load_model() is not None