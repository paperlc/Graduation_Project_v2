"""Core agent implementation with LLM, memory, vision defenses, and MCP tool calling."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import chromadb
import requests
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from .vision import verify_image_consistency
from src.utils.telemetry import new_trace_id, get_trace_id, set_trace_id, span


ToolCaller = Callable[..., Any]


class SimpleChatMemory:
    """Lightweight in-memory history container."""

    def __init__(self):
        self.chat_memory = self  # compatibility with older code paths
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
    trace: List[str]
    debug_messages: List[str]
    conversation_log: List[Dict[str, Any]]
    trace_id: str


class Web3Agent:
    """Web3 Agent with explicit Brain (LLM), Memory, Vision, and Tool modules."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        defense_enabled: bool | None = None,
        tool_caller: ToolCaller | None = None,
        monitored_accounts: Iterable[str] | None = None,
        chroma_path: str | None = None,
        mode: str = "chat",
        collection_name: str = "web3-rag",
    ):
        self.logger = logging.getLogger(__name__)
        llm_model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        llm_base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=llm_api_key,
            base_url=llm_base_url,
        )
        # Minimal in-memory history; can be replaced/disabled as needed.
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
        self.rag_tweet_file = os.getenv("RAG_TWEET_FILE") or (Path(__file__).resolve().parents[2] / "data" / "tweets.json")
        self.collection_name = collection_name

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
                name=self.collection_name, embedding_function=embedding_function
            )
            # seed optional tweet corpus if present
            try:
                tweet_path = Path(self.rag_tweet_file)
                if tweet_path.exists():
                    data = json.loads(tweet_path.read_text(encoding="utf-8"))
                    tweets = data.get("tweets", [])
                    if tweets:
                        ids = [t.get("id") or f"tweet-{i}" for i, t in enumerate(tweets)]
                        docs = [t.get("text", "") for t in tweets]
                        self.collection.upsert(ids=ids, documents=docs)
                        self.logger.info("Loaded %d tweets into RAG collection", len(tweets))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load tweet corpus: %s", exc)
        else:
            self.chroma_client = None

        # Build tool schemas for LLM function calling
        self.tools_schema = self._build_tools_schema()
        self.logger.info(
            "Agent initialized mode=%s defense=%s rag=%s vision=%s tools=%d",
            self.mode,
            self.defense_enabled,
            self.rag_enabled,
            self.vision_enabled,
            len(self.tools_schema),
        )

    def set_defense(self, enabled: bool) -> None:
        self.logger.info("Set defense mode -> %s", enabled)
        self.defense_enabled = enabled

    def set_mode(self, mode: str) -> None:
        """Switch work mode: chat | advisor."""
        normalized = (mode or "").lower()
        if normalized in {"advisor", "advice", "investment"}:
            self.mode = "advisor"
        else:
            self.mode = "chat"
        self.logger.info("Set agent mode -> %s", self.mode)

    def _build_tools_schema(self) -> List[Dict[str, Any]]:
        """Define tools available for LLM function calling."""
        tool_defs = [
            {"name": "get_eth_balance", "description": "Query ETH balance for an address.", "params": {"address": "string"}},
            {"name": "get_token_balance", "description": "Query token balance for an address.", "params": {"address": "string", "token_symbol": "string"}},
            {"name": "get_transaction_history", "description": "Get recent transactions for an address (for activity only, not ownership).", "params": {"address": "string", "limit": "integer"}},
            {"name": "get_contract_bytecode", "description": "Fetch contract bytecode.", "params": {"address": "string"}},
            {"name": "resolve_ens_domain", "description": "Resolve ENS domain to address.", "params": {"domain_name": "string"}},
            {"name": "get_token_price", "description": "Fetch token price from oracle.", "params": {"token_symbol": "string"}},
            {"name": "check_address_reputation", "description": "Check reputation/blacklist status.", "params": {"address": "string"}},
            {"name": "simulate_transaction", "description": "Simulate a transaction.", "params": {"to": "string", "value": "number", "data": "string"}},
            {"name": "verify_contract_owner", "description": "Return the owner/controller of a contract address (use when asked ‘谁是owner/主人/归属?’).", "params": {"contract_address": "string"}},
            {"name": "check_token_approval", "description": "Check allowance.", "params": {"owner": "string", "spender": "string"}},
            {"name": "verify_signature", "description": "Verify a signature.", "params": {"message": "string", "signature": "string", "address": "string"}},
            {"name": "transfer_eth", "description": "Send ETH.", "params": {"to_address": "string", "amount": "number", "sender": "string"}, "is_write": True},
            {"name": "swap_tokens", "description": "Simulate token swap.", "params": {"token_in": "string", "token_out": "string", "amount": "number", "address": "string"}, "is_write": True},
            {"name": "approve_token", "description": "Approve token spending.", "params": {"spender": "string", "amount": "number", "owner": "string"}, "is_write": True},
            {"name": "revoke_approval", "description": "Revoke token approval.", "params": {"spender": "string", "owner": "string"}, "is_write": True},
            {"name": "get_liquidity_pool_info", "description": "Query liquidity pool info.", "params": {"token_address": "string"}},
            {"name": "bridge_asset", "description": "Simulate bridge to another chain.", "params": {"token": "string", "target_chain": "string"}, "is_write": True},
            {"name": "stake_tokens", "description": "Simulate staking.", "params": {"protocol": "string", "amount": "number"}, "is_write": True},
        ]

        tools: List[Dict[str, Any]] = []
        for tool in tool_defs:
            params = tool["params"]
            properties = {k: {"type": v} for k, v in params.items()}
            required = list(params.keys())

            if tool.get("is_write"):
                # Optional idempotency token so safe/unsafe runs do not double-write.
                properties["idempotency_key"] = {
                    "type": "string",
                    "description": "Idempotency token to deduplicate write operations across runs.",
                }

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
        return tools

    def analyze_image(self, image_path: str, text_claim: str) -> bool:
        """Expose the vision module for external callers (e.g., UI)."""
        return verify_image_consistency(text_claim=text_claim, image_path=image_path)

    def add_knowledge(self, documents: List[str]) -> None:
        """Add documents into the local Chroma collection for RAG."""
        if not documents or not self.collection:
            return
        ids = [f"doc-{i}" for i in range(len(documents))]
        self.collection.upsert(ids=ids, documents=documents)
        self.logger.info("Added %d documents to RAG collection", len(documents))

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
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Remote RAG query failed: %s", exc)
            return ""

    def _query_rag(self, query: str) -> str:
        if not query or not self.rag_enabled:
            return ""
        if self.rag_provider == "remote":
            return self._query_rag_remote(query)
        return self._query_rag_local(query)

    def _call_tool(self, tool_name: str, **kwargs) -> Any:
        if not self.tool_caller:
            raise RuntimeError("tool_caller is not configured.")
        return self.tool_caller(tool_name, **kwargs)

    def _run_tool_calls(self, response: AIMessage, trace: List[str]) -> (AIMessage, List[BaseMessage]):
        """Execute LLM-requested tool calls and produce ToolMessage list."""
        tool_messages: List[BaseMessage] = []
        if not getattr(response, "tool_calls", None):
            return response, tool_messages

        for call in response.tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            try:
                result = self._call_tool(name, **args)
                result_str = json.dumps(result, ensure_ascii=False)
                trace.append(f"LLM tool {name}({args}) -> {result_str}")
                self.logger.info("tool_call success: %s args=%s result=%s", name, args, result_str)
            except Exception as exc:  # noqa: BLE001
                result_str = f"call {name} failed: {exc}"
                trace.append(result_str)
                self.logger.exception("tool_call failed: %s args=%s", name, args)
            tool_messages.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=call.get("id", ""),
                )
            )
        return response, tool_messages

    def _format_messages(self, messages: List[BaseMessage]) -> List[str]:
        formatted = []
        for msg in messages:
            role = msg.__class__.__name__
            content = getattr(msg, "content", "")
            if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"):
                tc = msg.tool_calls
                content = f"tool_calls={tc}"
            formatted.append(f"{role}: {content}")
        return formatted

    def _extract_accounts_from_text(self, text: str) -> List[str]:
        text_lower = text.lower()
        hits = [acc for acc in self.monitored_accounts if acc in text_lower]
        hits += re.findall(r"0x[a-fA-F0-9]{6,}", text)
        seen = set()
        unique: List[str] = []
        for h in hits:
            if h not in seen:
                unique.append(h)
                seen.add(h)
        return unique

    def _gather_chain_context(self, accounts: List[str]) -> str:
        if not accounts or not self.tool_caller:
            return ""

        snapshots = []
        for account in accounts:
            try:
                try:
                    balance = self._call_tool("get_eth_balance", address=account)
                except Exception:
                    balance = self._call_tool("get_balance", account=account)
                snapshots.append(f"{account}: {balance}")
            except Exception as exc:  # noqa: BLE001
                snapshots.append(f"{account}: error={exc}")

        return "\n".join(snapshots)

    def _build_system_prompt(self, chain_context: str, rag_context: str, vision_note: str) -> str:
        if self.mode == "advisor":
            base = (
                "You are Web3Agent in advisor mode. Speak in concise, natural Chinese. "
                "先给出结论，再补充理由与风险。语气友好，不要机械或重复。"
                "Use on-chain data, retrieved intel, and visual checks to validate claims. "
                "默认把链上快照当作真实主网，不必解释“模拟”与否。回答用纯文本，不要输出 XML/DSML/JSON 片段。"
            )
        else:
            base = (
                "You are Web3Agent in dialogue mode. Speak in concise, natural Chinese. "
                "先说答案，再补充关键依据。语气友好且口语化，不要机械。"
                "Use provided on-chain data and retrieved intel; treat snapshots as真实主网，无需解释“模拟”。"
                "回答用纯文本，不要输出 XML/DSML/JSON 片段。"
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
        trace_id = get_trace_id() or new_trace_id()
        set_trace_id(trace_id)
        self.logger.info(
            "chat start mode=%s defense=%s input=%s image=%s",
            self.mode,
            self.defense_enabled,
            user_input,
            bool(image),
        )
        vision_checked = False
        vision_consistent: Optional[bool] = None
        trace: List[str] = []
        conversation_log: List[Dict[str, Any]] = []

        if self.defense_enabled and self.vision_enabled and image:
            vision_checked = True
            vision_consistent = self.analyze_image(image, user_input)
            trace.append(f"Vision check: {'PASS' if vision_consistent else 'FAIL'}")

        rag_context = ""
        if self.defense_enabled and self.rag_enabled:
            rag_context = self._query_rag(user_input)
            trace.append("RAG query executed" if rag_context else "RAG query empty")

        chain_context = ""
        target_accounts: List[str] = []
        if self.defense_enabled:
            target_accounts = self.monitored_accounts
        else:
            target_accounts = self._extract_accounts_from_text(user_input)

        if target_accounts:
            chain_context = self._gather_chain_context(target_accounts)
            trace.append(f"Chain snapshot: {', '.join(target_accounts)}")
        else:
            trace.append("Chain snapshot skipped (no accounts or defense off)")

        if not self.defense_enabled:
            trace.append("Defense disabled: no auto chain snapshot/vision/RAG")

        vision_note = ""
        if vision_checked:
            status = "PASS" if vision_consistent else "FAIL"
            vision_note = f"Vision consistency check: {status}"

        system_prompt = self._build_system_prompt(chain_context, rag_context, vision_note)

        history_messages = self.memory.load()
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt), *history_messages, HumanMessage(content=user_input)]
        self.logger.debug("messages before LLM: %s", [m.__class__.__name__ for m in messages])
        self.logger.debug("system prompt: %s", system_prompt)

        max_rounds = int(os.getenv("TOOL_CALL_MAX_ROUNDS", "3"))
        round_idx = 1
        with span("chat", {"trace_id": trace_id, "mode": self.mode, "defense": self.defense_enabled}):
            while True:
                conversation_log.append({"label": f"LLM call #{round_idx} input", "messages": self._format_messages(messages)})

                with span(f"llm_call_{round_idx}", {"trace_id": trace_id}):
                    response = self.llm.invoke(messages, tools=self.tools_schema, tool_choice="auto")
                conversation_log.append(
                    {
                        "label": f"LLM call #{round_idx} output",
                        "messages": [f"AIMessage: {getattr(response, 'content', '')}"],
                        "tool_calls": getattr(response, "tool_calls", None),
                    }
                )

                if not getattr(response, "tool_calls", None):
                    break

                if round_idx >= max_rounds:
                    trace.append(f"Max tool call rounds reached ({max_rounds}); stopping.")
                    break

                with span(f"tool_exec_round_{round_idx}", {"trace_id": trace_id}):
                    _, tool_messages = self._run_tool_calls(response, trace)
                messages.extend([response, *tool_messages])
                self.logger.debug("messages after tools round %d: %s", round_idx, [m.__class__.__name__ for m in messages])
                round_idx += 1

        reply_text = response.content if isinstance(response, AIMessage) else str(response)

        debug_messages = [f"{m.__class__.__name__}: {getattr(m, 'content', '')}" for m in messages]
        debug_messages.append(f"AI: {reply_text}")

        self.memory.add_user_message(user_input)
        self.memory.add_ai_message(reply_text)
        self.logger.info("chat end, reply length=%d trace=%s", len(reply_text), trace)

        return ChatResult(
            reply=reply_text,
            vision_checked=vision_checked,
            vision_consistent=vision_consistent,
            chain_context=chain_context or None,
            rag_context=rag_context or None,
            trace=trace,
            debug_messages=debug_messages,
            conversation_log=conversation_log,
            trace_id=trace_id,
        )
