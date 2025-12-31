"""Core agent implementation with LLM, memory, and vision defenses."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import chromadb
import requests
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .vision import verify_image_consistency


ToolCaller = Callable[[str, Dict[str, Any]], Any]


class SimpleChatMemory:
    """轻量自定义记忆，兼容注入与清空。"""

    def __init__(self):
        self.chat_memory = self  # 兼容旧的注入代码
        self._messages: List[BaseMessage] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str) -> None:
        self._messages.append(AIMessage(content=text))

    def load(self) -> List[BaseMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


@dataclass
class ChatResult:
    reply: str
    vision_checked: bool
    vision_consistent: Optional[bool]
    chain_context: Optional[str]
    rag_context: Optional[str]


class Web3Agent:
    """Web3 Agent with explicit Brain (LLM), Memory, and Vision modules."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        defense_enabled: bool | None = None,
        tool_caller: ToolCaller | None = None,
        monitored_accounts: Iterable[str] | None = None,
        chroma_path: str | None = None,
        mode: str = "chat",
    ):
        llm_model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        llm_base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=llm_api_key,
            base_url=llm_base_url,
        )
        self.memory = SimpleChatMemory()
        defense_default = (
            defense_enabled
            if defense_enabled is not None
            else (os.getenv("DEFENSE_DEFAULT_ON", "true").lower() != "false")
        )
        self.defense_enabled = bool(defense_default)
        self.tool_caller = tool_caller
        self.monitored_accounts = list(monitored_accounts or ["treasury", "alice", "bob", "charlie"])
        self.mode = mode  # chat | advisor
        self.rag_provider = (os.getenv("RAG_PROVIDER") or "local").lower()
        self.rag_remote_url = os.getenv("RAG_REMOTE_URL")
        self.rag_remote_api_key = os.getenv("RAG_REMOTE_API_KEY")
        self.rag_enabled = self.rag_provider != "off"
        self.vision_enabled = (os.getenv("VISION_ENABLED", "true").lower() != "false")

        # Simple ChromaDB setup for RAG; keeps it local and replaceable.
        embedding_model = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
        embedding_api_key = os.getenv("EMBEDDING_API_KEY") or llm_api_key
        embedding_api_base = os.getenv("EMBEDDING_API_BASE") or llm_base_url or "https://api.openai.com/v1"
        chroma_dir = chroma_path or os.getenv("CHROMA_PATH")

        self.collection = None
        if self.rag_enabled and self.rag_provider == "local" and embedding_api_key:
            embedding_function = OpenAIEmbeddingFunction(
                api_key=embedding_api_key,
                model_name=embedding_model,
                api_base=embedding_api_base,
            )
            self.chroma_client = chromadb.Client(
                Settings(
                    is_persistent=bool(chroma_dir),
                    persist_directory=chroma_dir,
                    anonymized_telemetry=False,
                )
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="web3-rag", embedding_function=embedding_function
            )
        else:
            self.chroma_client = None

    def set_defense(self, enabled: bool) -> None:
        self.defense_enabled = enabled

    def set_mode(self, mode: str) -> None:
        """切换工作模式：chat（对话）或 advisor（顾问/建议）。"""
        normalized = (mode or "").lower()
        if normalized in {"advisor", "advice", "investment"}:
            self.mode = "advisor"
        else:
            self.mode = "chat"

    def analyze_image(self, image_path: str, text_claim: str) -> bool:
        """Expose the vision module for external callers (e.g., UI)."""
        return verify_image_consistency(text_claim=text_claim, image_path=image_path)

    def add_knowledge(self, documents: List[str]) -> None:
        """Add documents into the local Chroma collection for RAG."""
        if not documents or not self.collection:
            return
        ids = [f"doc-{i}" for i in range(len(documents))]
        self.collection.upsert(ids=ids, documents=documents)

    def _query_rag_local(self, query: str) -> str:
        if not self.collection:
            return ""
        results = self.collection.query(query_texts=[query], n_results=3)
        docs = results.get("documents") or []
        flattened = docs[0] if docs else []
        return "\n".join(flattened)

    def _query_rag_remote(self, query: str) -> str:
        if not self.rag_remote_url:
            return ""
        try:
            headers = {"Content-Type": "application/json"}
            if self.rag_remote_api_key:
                headers["Authorization"] = f"Bearer {self.rag_remote_api_key}"
            payload = {"query": query, "top_k": 3}
            resp = requests.post(self.rag_remote_url, json=payload, timeout=10, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("documents") or data.get("results") or []
            if docs and isinstance(docs[0], dict) and "text" in docs[0]:
                docs = [d.get("text", "") for d in docs]
            return "\n".join(docs)
        except Exception:
            return ""

    def _query_rag(self, query: str) -> str:
        if not query:
            return ""
        if not self.rag_enabled:
            return ""
        if self.rag_provider == "remote":
            return self._query_rag_remote(query)
        return self._query_rag_local(query)

    def _call_tool(self, tool_name: str, **kwargs) -> Any:
        if not self.tool_caller:
            return None
        return self.tool_caller(tool_name, **kwargs)

    def _gather_chain_context(self) -> str:
        if not self.defense_enabled or not self.tool_caller:
            return ""

        snapshots = []
        for account in self.monitored_accounts:
            try:
                try:
                    balance = self._call_tool("get_eth_balance", address=account)
                except Exception:
                    balance = self._call_tool("get_balance", account=account)
                snapshots.append(f"{account}: {balance}")
            except Exception as exc:  # best-effort defense, do not crash
                snapshots.append(f"{account}: error={exc}")

        return "\n".join(snapshots)

    def _build_system_prompt(self, chain_context: str, rag_context: str, vision_note: str) -> str:
        if self.mode == "advisor":
            base = (
                "You are Web3Agent in advisor mode. Provide cautious, risk-aware investment and security suggestions. "
                "Use on-chain data, retrieved intel, and visual checks to validate claims. "
                "If data is missing, state assumptions and avoid overconfident actions."
            )
        else:
            base = (
                "You are Web3Agent in dialogue mode. Be helpful, concise, and factual. "
                "Use provided on-chain data and retrieved intel when available. Flag suspicious or inconsistent content."
            )
        context_blocks = []
        if chain_context:
            context_blocks.append(f"On-chain snapshot:\n{chain_context}")
        if rag_context:
            context_blocks.append(f"Retrieved intel:\n{rag_context}")
        if vision_note:
            context_blocks.append(vision_note)

        context_blob = "\n\n".join(context_blocks)
        if context_blob:
            return f"{base}\n\n{context_blob}"
        return base

    def chat(self, user_input: str, image: str | None = None) -> ChatResult:
        """Main chat entrypoint."""
        vision_checked = False
        vision_consistent: Optional[bool] = None

        if self.defense_enabled and self.vision_enabled and image:
            vision_checked = True
            vision_consistent = self.analyze_image(image, user_input)

        rag_context = self._query_rag(user_input) if (self.defense_enabled and self.rag_enabled) else ""
        chain_context = self._gather_chain_context() if self.defense_enabled else ""

        vision_note = ""
        if vision_checked:
            status = "PASS" if vision_consistent else "FAIL"
            vision_note = f"Vision consistency check: {status}"

        system_prompt = self._build_system_prompt(chain_context, rag_context, vision_note)

        history_messages = self.memory.load()
        messages = [SystemMessage(content=system_prompt), *history_messages, HumanMessage(content=user_input)]

        response = self.llm.invoke(messages)
        reply_text = response.content if isinstance(response, AIMessage) else str(response)

        # Keep memory open for external manipulation (memory injection attack surface).
        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(reply_text)

        return ChatResult(
            reply=reply_text,
            vision_checked=vision_checked,
            vision_consistent=vision_consistent,
            chain_context=chain_context or None,
            rag_context=rag_context or None,
        )
