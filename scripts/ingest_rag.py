#!/usr/bin/env python
"""
Ingest local documents into Chroma for RAG.

Usage:
  python scripts/ingest_rag.py --src data/rag

Env vars (fallbacks):
  EMBEDDING_API_KEY / EMBEDDING_API_BASE / EMBEDDING_MODEL
  CHROMA_PATH
  RAG_COLLECTION_SAFE (default: web3-rag-safe)
  RAG_COLLECTION_UNSAFE (default: web3-rag-unsafe)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv


def load_texts(src: Path) -> List[Tuple[str, str]]:
    """Load .md/.txt files under src, returning (id, content)."""
    docs: List[Tuple[str, str]] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        doc_id = path.relative_to(src).as_posix()
        docs.append((doc_id, text))
    return docs


def load_tweets(tweet_file: Path) -> List[Tuple[str, str]]:
    """Load tweet texts from a JSON file shaped like {"tweets":[{"id":..,"text":..}, ...]}."""
    if not tweet_file.exists():
        return []
    data = json.loads(tweet_file.read_text(encoding="utf-8"))
    tweets = data.get("tweets", [])
    docs: List[Tuple[str, str]] = []
    for i, t in enumerate(tweets):
        tid = t.get("id") or f"tweet-{i}"
        text = t.get("text", "")
        if text:
            docs.append((f"tweet/{tid}", text))
    return docs


def ingest(docs: List[Tuple[str, str]], collection_name: str, chroma_dir: str, embedding_fn: OpenAIEmbeddingFunction) -> None:
    client = chromadb.Client(
        Settings(
            is_persistent=bool(chroma_dir),
            persist_directory=chroma_dir,
            anonymized_telemetry=False,
        )
    )
    if RESET_COLLECTIONS:
        try:
            client.delete_collection(name=collection_name)
            print(f"[info] Dropped existing collection {collection_name} before ingest.")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Failed to drop collection {collection_name}: {exc}")
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)
    ids = [doc_id for doc_id, _ in docs]
    texts = [text for _, text in docs]
    if not ids:
        print(f"[info] No documents to ingest for {collection_name}")
        return
    print(f"[info] Ingesting {len(ids)} docs into collection={collection_name}")
    collection.upsert(ids=ids, documents=texts)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest local documents into Chroma.")
    parser.add_argument("--src", type=Path, default=Path("data/rag"), help="Directory containing .md/.txt files.")
    parser.add_argument("--tweets", type=Path, default=Path("data/tweets.json"), help="Tweet JSON file to include.")
    parser.add_argument("--skip-tweets", action="store_true", help="Do not ingest tweets.")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset collections before ingest.")
    args = parser.parse_args()

    chroma_dir = os.getenv("CHROMA_PATH") or ""
    primary_local_model = os.getenv("EMBEDDING_LOCAL_MODEL")
    local_model_list = os.getenv("EMBEDDING_LOCAL_MODELS", "")
    embedding_local_models = []
    if primary_local_model:
        embedding_local_models.append(primary_local_model)
    if local_model_list:
        for item in local_model_list.split(","):
            name = item.strip()
            if name and name not in embedding_local_models:
                embedding_local_models.append(name)
    if not embedding_local_models:
        embedding_local_models = ["sentence-transformers/all-MiniLM-L6-v2"]
    embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    embedding_model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
    embedding_api_base = os.getenv("EMBEDDING_API_BASE") or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1"
    embedding_use_local = os.getenv("EMBEDDING_USE_LOCAL", "true").lower() != "false"
    embedding_use_remote = os.getenv("EMBEDDING_USE_REMOTE", "false").lower() == "true"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    global RESET_COLLECTIONS
    reset_env = os.getenv("RAG_RESET_COLLECTIONS", "true").lower()
    RESET_COLLECTIONS = not args.no_reset and reset_env != "false"

    emb_fn = None
    if embedding_use_local and embedding_local_models:
        for candidate in embedding_local_models:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(candidate, token=hf_token)

                class LocalEmbeddingFunction:
                    def name(self):
                        return f"local-st-{candidate}"

                    def _normalize(self, inp):
                        if isinstance(inp, str):
                            texts = [inp]
                        elif isinstance(inp, list):
                            texts = []
                            for item in inp:
                                if isinstance(item, str):
                                    texts.append(item)
                                elif isinstance(item, (tuple, list)) and item:
                                    texts.append(str(item[0]))
                                else:
                                    texts.append(str(item))
                        else:
                            texts = [str(inp)]
                        return [t for t in texts if t]

                    def __call__(self, input):  # backward compat
                        texts = self._normalize(input)
                        return model.encode(texts, convert_to_numpy=True).tolist() if texts else []

                    def embed_documents(self, input):
                        texts = self._normalize(input)
                        return model.encode(texts, convert_to_numpy=True).tolist() if texts else []

                    def embed_query(self, input):
                        texts = self._normalize(input)
                        return model.encode(texts, convert_to_numpy=True).tolist() if texts else []

                emb_fn = LocalEmbeddingFunction()
                print(f"[info] Using local embedding model: {candidate}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] Failed to load local embedding model {candidate}: {exc}")
    if emb_fn is None and embedding_use_remote:
        if not embedding_api_key:
            raise SystemExit("Need EMBEDDING_API_KEY/LLM_API_KEY or EMBEDDING_LOCAL_MODEL for embeddings.")
        emb_fn = OpenAIEmbeddingFunction(
            api_key=embedding_api_key,
            model_name=embedding_model,
            api_base=embedding_api_base,
        )
        print(f"[info] Using OpenAI-compatible embedding: {embedding_model}")
    if emb_fn is None:
        raise SystemExit("No embedding function available: enable EMBEDDING_USE_LOCAL or EMBEDDING_USE_REMOTE with proper config.")

    docs = load_texts(args.src)
    if not args.skip_tweets:
        docs.extend(load_tweets(args.tweets))

    col_safe = os.getenv("RAG_COLLECTION_SAFE") or "web3-rag-safe"
    col_unsafe = os.getenv("RAG_COLLECTION_UNSAFE") or "web3-rag-unsafe"
    ingest(docs, col_safe, chroma_dir, emb_fn)
    ingest(docs, col_unsafe, chroma_dir, emb_fn)
    print("[done] RAG ingestion finished.")


if __name__ == "__main__":
    main()
