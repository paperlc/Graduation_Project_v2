"""RAG poisoning attack utilities."""

from __future__ import annotations

import uuid
from typing import Any


def poison_rag(collection: Any, text: str) -> str:
    """
    Insert malicious content directly into a ChromaDB collection.

    Returns the inserted document id so callers can trace the injection.
    """
    if not hasattr(collection, "upsert"):
        raise AttributeError("Collection does not support upsert.")

    doc_id = f"poison-{uuid.uuid4()}"
    collection.upsert(ids=[doc_id], documents=[text])
    return doc_id
