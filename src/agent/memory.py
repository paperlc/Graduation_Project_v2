"""Persistent chat memory with ChromaDB backend (Global Shared Architecture).

This module provides ChromaChatMemory with:
- Global shared storage across sessions (not per-session isolation)
- Access count tracking for memory cleanup
- Automatic cleanup based on access frequency and time
- Independent storage directory ./data/memory/
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ChromaChatMemory:
    """
    ChromaDB-backed persistent chat memory with global shared storage.

    Architecture changes from per-session isolation:
    - Collection naming: web3-memory-{lane} (shared) or web3-memory (fully shared)
    - Document ID format: {session_id}_{message_number:03d}_{role}_{timestamp}
    - Messages are filtered by session_id metadata during load/search

    Features:
    - Persistent storage across sessions
    - Semantic search across conversation history
    - CRUD operations on messages
    - Access count tracking for cleanup
    - Automatic cleanup of old/unaccessed messages
    """

    def __init__(
        self,
        lane: str = "safe",
        chroma_path: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        shared_mode: str = "lane",
        cleanup_mode: str = "startup",
        retention_days: int = 10,
    ):
        """
        Initialize global shared memory with ChromaDB backend.

        Args:
            lane: "safe" or "unsafe" lane identifier
            chroma_path: Path to ChromaDB storage (default: ./data/memory)
            embedding_function: Embedding function for vectorization
            shared_mode: "lane" (lane-isolated) or "global" (fully shared)
            cleanup_mode: "startup" | "scheduled" | "manual"
            retention_days: Days to retain messages before cleanup
        """
        self.lane = lane
        self.shared_mode = shared_mode
        self.cleanup_mode = cleanup_mode
        self.retention_days = retention_days
        self._messages: List[BaseMessage] = []
        self._current_session_id: Optional[str] = None
        self._message_counts: Dict[str, int] = {}  # Track message count per session

        # Debug output
        print(f"[Memory] Initializing with shared_mode={shared_mode}, lane={lane}", file=sys.stderr)
        sys.stderr.flush()

        # Determine storage path
        if chroma_path is None:
            chroma_path = os.getenv("MEMORY_PATH", "./data/memory")

        print(f"[Memory] Using chroma_path={chroma_path}", file=sys.stderr)
        sys.stderr.flush()

        # Create independent ChromaDB client (separate from RAG)
        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        logger.info("Created independent ChromaDB client for memory at %s", chroma_path)

        self.embedding_function = embedding_function

        # Determine collection name based on shared mode
        if shared_mode == "global":
            self.collection_name = "web3-memory"
        else:  # "lane" (default)
            self.collection_name = f"web3-memory-{lane}"

        # Get or create collection
        try:
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"lane": lane, "type": "memory", "shared_mode": shared_mode},
            )
            logger.info("Using collection: %s (mode: %s)", self.collection_name, shared_mode)
        except Exception as exc:
            logger.error("Failed to get/create collection %s: %s", self.collection_name, exc)
            raise

        # Run cleanup on startup if configured
        if cleanup_mode == "startup":
            try:
                deleted = self.cleanup()
                if deleted > 0:
                    logger.info("Startup cleanup: removed %d old messages", deleted)
            except Exception as exc:
                logger.warning("Startup cleanup failed: %s", exc)

    # ===== Core Interface =====

    def add_user_message(
        self, text: str, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a user message to memory.

        Args:
            text: Message text content
            session_id: Session identifier this message belongs to
            metadata: Optional metadata to attach to the message
        """
        msg_num = self._get_next_message_number(session_id)
        doc_id = self._generate_doc_id(session_id, msg_num, "user")

        # Build metadata with access tracking
        now_ts = int(time.time())
        meta = {
            "session_id": session_id,
            "message_number": msg_num,
            "role": "user",
            "lane": self.lane,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unix_timestamp": now_ts,
            "access_count": 0,
            "last_accessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_accessed_ts": now_ts,
            **(metadata or {}),
        }

        # Add to in-memory list if current session
        if self._current_session_id == session_id:
            self._messages.append(HumanMessage(content=text))

        # Persist to ChromaDB
        try:
            self._collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
            logger.debug("Added user message %d for session %s", msg_num, session_id)
        except Exception as exc:
            logger.error("Failed to add user message: %s", exc)

    def add_ai_message(
        self, text: str, session_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an AI message to memory.

        Args:
            text: Message text content
            session_id: Session identifier this message belongs to
            metadata: Optional metadata to attach to the message
        """
        msg_num = self._get_next_message_number(session_id)
        doc_id = self._generate_doc_id(session_id, msg_num, "assistant")

        # Build metadata with access tracking
        now_ts = int(time.time())
        meta = {
            "session_id": session_id,
            "message_number": msg_num,
            "role": "assistant",
            "lane": self.lane,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "unix_timestamp": now_ts,
            "access_count": 0,
            "last_accessed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "last_accessed_ts": now_ts,
            **(metadata or {}),
        }

        # Add to in-memory list if current session
        if self._current_session_id == session_id:
            self._messages.append(AIMessage(content=text))

        # Persist to ChromaDB
        try:
            self._collection.upsert(ids=[doc_id], documents=[text], metadatas=[meta])
            logger.debug("Added AI message %d for session %s", msg_num, session_id)
        except Exception as exc:
            logger.error("Failed to add AI message: %s", exc)

    def load(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Load messages for a specific session from memory.

        Args:
            session_id: Session identifier to load messages for
            limit: Maximum number of recent messages to load (None = all)

        Returns:
            List of LangChain BaseMessage objects in chronological order
        """
        # Update current session
        self._current_session_id = session_id

        # Build filter based on shared mode
        # In global mode, load all messages (no session_id filter)
        # In lane mode, load only this session's messages
        if self.shared_mode == "global":
            where = {"lane": self.lane}
            logger.info("Loading messages in GLOBAL mode (lane=%s, no session filter)", self.lane)
        else:  # "lane" mode - filter by session_id
            where = {"$and": [{"session_id": session_id}, {"lane": self.lane}]}
            logger.info("Loading messages in LANE mode (session_id=%s, lane=%s)", session_id, self.lane)

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"]
            )

            self._messages.clear()

            if results["documents"]:
                # Sort by message_number
                docs_with_meta = list(zip(results["documents"], results["metadatas"]))
                sorted_docs = sorted(docs_with_meta, key=lambda x: x[1].get("message_number", 0))

                for doc, meta in sorted_docs:
                    role = meta.get("role", "user")
                    if role == "user":
                        self._messages.append(HumanMessage(content=doc))
                    elif role == "assistant":
                        self._messages.append(AIMessage(content=doc))
                    elif role == "system":
                        self._messages.append(SystemMessage(content=doc))

            logger.info("Loaded %d messages for session %s (shared_mode=%s)", len(self._messages), session_id, self.shared_mode)

            if limit is None or limit >= len(self._messages):
                return list(self._messages)
            return self._messages[-limit:]

        except Exception as exc:
            logger.error("Failed to load messages for session %s: %s", session_id, exc)
            return []

    def clear(self, session_id: str) -> None:
        """
        Clear all messages for a specific session.

        Args:
            session_id: Session identifier to clear
        """
        # Clear in-memory cache if this is the current session
        if self._current_session_id == session_id:
            self._messages.clear()
            self._current_session_id = None

        # Remove message count for this session
        if session_id in self._message_counts:
            del self._message_counts[session_id]

        # Delete all messages for this session
        try:
            # Get all document IDs for this session using $and operator
            results = self._collection.get(
                where={"$and": [{"session_id": session_id}, {"lane": self.lane}]},
                include=["documents"]
            )

            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info("Cleared %d messages for session %s", len(results["ids"]), session_id)
        except Exception as exc:
            logger.error("Failed to clear session %s: %s", session_id, exc)

    # ===== Extended CRUD Operations =====

    def delete_message(self, message_id: str) -> bool:
        """
        Delete a specific message by its document ID.

        Args:
            message_id: The document ID to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            self._collection.delete(ids=[message_id])
            # Reload current session messages
            if self._current_session_id:
                self.load(self._current_session_id)
            logger.info("Deleted message %s", message_id)
            return True
        except Exception as exc:
            logger.error("Failed to delete message %s: %s", message_id, exc)
            return False

    def edit_message(self, message_id: str, new_text: str) -> bool:
        """
        Edit a specific message's content.

        Args:
            message_id: The document ID to edit
            new_text: New message content

        Returns:
            True if edited, False otherwise
        """
        try:
            results = self._collection.get(ids=[message_id], include=["metadatas"])
            if results["ids"]:
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                # Update timestamps
                metadata["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                metadata["unix_timestamp"] = int(time.time())

                self._collection.update(ids=[message_id], documents=[new_text], metadatas=[metadata])
                # Reload current session messages
                if self._current_session_id:
                    self.load(self._current_session_id)
                logger.info("Edited message %s", message_id)
                return True
        except Exception as exc:
            logger.error("Failed to edit message %s: %s", message_id, exc)
        return False

    def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific message.

        Args:
            message_id: The document ID to retrieve

        Returns:
            Dictionary with 'content' and 'metadata' keys, or None if not found
        """
        try:
            results = self._collection.get(ids=[message_id], include=["documents", "metadatas"])
            if results["ids"]:
                return {
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                }
        except Exception as exc:
            logger.error("Failed to get message %s: %s", message_id, exc)
        return None

    # ===== Search Operations (with access tracking) =====

    def search_messages(
        self,
        query: str,
        n_results: int = 5,
        role_filter: Optional[str] = None,
        update_access: bool = True,
        lane_filter: Optional[str] = None,
        session_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across messages.

        Args:
            query: Search query text
            n_results: Number of results to return
            role_filter: Optional "user" or "assistant" filter
            update_access: Whether to update access counts (default: True)
            lane_filter: Optional lane filter
            session_filter: Optional session ID filter

        Returns:
            List of {content, metadata, distance} dicts
        """
        if not self.embedding_function:
            logger.warning("Cannot search: no embedding function available")
            return []

        # Build filter conditions using $and operator
        conditions = [{"lane": lane_filter or self.lane}]
        if session_filter:
            conditions.append({"session_id": session_filter})
        if role_filter:
            conditions.append({"role": role_filter})

        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            messages = []
            message_ids = []

            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    messages.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    })
                    message_ids.append(results["ids"][0][i])

            # Update access counts if enabled
            if update_access and message_ids:
                self.batch_increment_access(message_ids)

            return messages
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            return []

    # ===== Access Count Tracking =====

    def increment_access(self, message_id: str) -> None:
        """
        Increment access count for a single message.

        Args:
            message_id: The document ID to update
        """
        try:
            results = self._collection.get(ids=[message_id], include=["metadatas"])
            if results["ids"] and results["metadatas"]:
                metadata = results["metadatas"][0]
                current_count = metadata.get("access_count", 0)

                # Update metadata
                metadata["access_count"] = current_count + 1
                metadata["last_accessed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                metadata["last_accessed_ts"] = int(time.time())

                self._collection.update(ids=[message_id], metadatas=[metadata])
                logger.debug("Incremented access count for %s", message_id)
        except Exception as exc:
            logger.error("Failed to increment access for %s: %s", message_id, exc)

    def batch_increment_access(self, message_ids: List[str]) -> None:
        """
        Increment access count for multiple messages.

        Args:
            message_ids: List of document IDs to update
        """
        if not message_ids:
            return

        try:
            results = self._collection.get(ids=message_ids, include=["metadatas"])

            if not results["ids"]:
                return

            metadatas = []
            for i, doc_id in enumerate(results["ids"]):
                if results["metadatas"] and i < len(results["metadatas"]):
                    metadata = results["metadatas"][i]
                    current_count = metadata.get("access_count", 0)

                    # Update metadata
                    metadata["access_count"] = current_count + 1
                    metadata["last_accessed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    metadata["last_accessed_ts"] = int(time.time())

                    metadatas.append(metadata)

            if metadatas:
                self._collection.update(ids=results["ids"], metadatas=metadatas)
                logger.debug("Incremented access count for %d messages", len(message_ids))
        except Exception as exc:
            logger.error("Failed to batch increment access: %s", exc)

    def get_access_stats(self) -> Dict[str, Any]:
        """
        Get access statistics for all messages.

        Returns:
            Dictionary with statistics
        """
        try:
            results = self._collection.get(include=["metadatas"])

            stats = {
                "total_messages": len(results["ids"]) if results["ids"] else 0,
                "access_counts": {},
                "last_accessed": {},
            }

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    session_id = metadata.get("session_id", "unknown")
                    access_count = metadata.get("access_count", 0)
                    last_accessed = metadata.get("last_accessed_at", "never")

                    # Per-session stats
                    if session_id not in stats["access_counts"]:
                        stats["access_counts"][session_id] = 0
                        stats["last_accessed"][session_id] = last_accessed

                    stats["access_counts"][session_id] += access_count

                    # Update last accessed if more recent
                    if last_accessed != "never" and last_accessed > stats["last_accessed"][session_id]:
                        stats["last_accessed"][session_id] = last_accessed

            return stats
        except Exception as exc:
            logger.error("Failed to get access stats: %s", exc)
            return {}

    # ===== Cleanup Mechanism =====

    def cleanup(
        self,
        retention_days: Optional[int] = None,
        min_access_count: int = 1
    ) -> int:
        """
        Cleanup old messages based on access frequency and time.

        Args:
            retention_days: Days to retain (None = use instance default)
            min_access_count: Minimum access count to protect

        Returns:
            Number of messages deleted
        """
        retention_days = retention_days or self.retention_days
        cutoff_ts = int(time.time()) - (retention_days * 86400)

        try:
            # Get all documents
            all_docs = self._collection.get(include=["metadatas"])

            if not all_docs["ids"]:
                return 0

            ids_to_delete = []
            for i, metadata in enumerate(all_docs["metadatas"]):
                # Use last_accessed_ts if available, otherwise fall back to unix_timestamp
                last_accessed = metadata.get(
                    "last_accessed_ts",
                    metadata.get("unix_timestamp", 0)
                )
                access_count = metadata.get("access_count", 0)

                # Delete if: older than cutoff AND access count below threshold
                if last_accessed < cutoff_ts and access_count < min_access_count:
                    ids_to_delete.append(all_docs["ids"][i])

            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info(
                    "Cleanup: deleted %d messages (older than %d days, access < %d)",
                    len(ids_to_delete),
                    retention_days,
                    min_access_count
                )

                # Reload current session if needed
                if self._current_session_id:
                    self.load(self._current_session_id)

            return len(ids_to_delete)

        except Exception as exc:
            logger.error("Cleanup failed: %s", exc)
            return 0

    # ===== Session Management =====

    def list_sessions(self, lane: Optional[str] = None) -> List[str]:
        """
        List all unique session IDs.

        Args:
            lane: Optional lane filter ("safe" or "unsafe")

        Returns:
            List of session IDs
        """
        try:
            results = self._collection.get(include=["metadatas"])
            sessions = set()

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    metadata_lane = metadata.get("lane", self.lane)
                    if lane is None or metadata_lane == lane:
                        session_id = metadata.get("session_id")
                        if session_id:
                            sessions.add(session_id)

            return sorted(list(sessions))
        except Exception as exc:
            logger.error("Failed to list sessions: %s", exc)
            return []

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        try:
            results = self._collection.get(
                where={"$and": [{"session_id": session_id}, {"lane": self.lane}]},
                include=["metadatas"]
            )

            stats = {
                "session_id": session_id,
                "message_count": 0,
                "user_messages": 0,
                "ai_messages": 0,
                "total_access_count": 0,
                "created_at": None,
                "last_accessed_at": None,
            }

            if results["metadatas"]:
                stats["message_count"] = len(results["metadatas"])

                for metadata in results["metadatas"]:
                    role = metadata.get("role", "")
                    if role == "user":
                        stats["user_messages"] += 1
                    elif role == "assistant":
                        stats["ai_messages"] += 1

                    stats["total_access_count"] += metadata.get("access_count", 0)

                    created_at = metadata.get("created_at")
                    if created_at and (stats["created_at"] is None or created_at < stats["created_at"]):
                        stats["created_at"] = created_at

                    last_accessed = metadata.get("last_accessed_at")
                    if last_accessed and (stats["last_accessed_at"] is None or last_accessed > stats["last_accessed_at"]):
                        stats["last_accessed_at"] = last_accessed

            return stats
        except Exception as exc:
            logger.error("Failed to get session stats for %s: %s", session_id, exc)
            return {}

    # ===== Compatibility Attributes =====

    @property
    def chat_memory(self):
        """Compatibility with code that expects memory.chat_memory."""
        return self

    # ===== Private Helper Methods =====

    def _generate_doc_id(self, session_id: str, message_number: int, role: str) -> str:
        """Generate document ID from session_id, message number and role."""
        timestamp = int(time.time())
        role_map = {"user": "user", "assistant": "assistant", "system": "system"}
        role_part = role_map.get(role, role)
        return f"{session_id}_{message_number:03d}_{role_part}_{timestamp}"

    def _get_next_message_number(self, session_id: str) -> int:
        """Get next message number for a session."""
        if session_id not in self._message_counts:
            # Load current count from database
            self._message_counts[session_id] = self._load_message_count(session_id)

        self._message_counts[session_id] += 1
        return self._message_counts[session_id]

    def _load_message_count(self, session_id: str) -> int:
        """Load current message count for a session from collection."""
        try:
            results = self._collection.get(
                where={"$and": [{"session_id": session_id}, {"lane": self.lane}]},
                include=["metadatas"]
            )
            if results["metadatas"]:
                max_num = max(m.get("message_number", 0) for m in results["metadatas"])
                return max_num
        except Exception as exc:
            logger.warning("Failed to load message count for session %s: %s", session_id, exc)
        return 0


# For backward compatibility - SimpleChatMemory is now ChromaChatMemory
SimpleChatMemory = ChromaChatMemory
