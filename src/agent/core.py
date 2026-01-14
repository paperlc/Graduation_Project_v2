"""Core agent implementation with LLM, memory, vision defenses, and MCP tool calling."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_temperature = temperature
        self._openai_client = None

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
        # Embedding strategy switches
        primary_local_model = os.getenv("EMBEDDING_LOCAL_MODEL")
        local_model_list = os.getenv("EMBEDDING_LOCAL_MODELS", "")
        embedding_local_models: List[str] = []
        if primary_local_model:
            embedding_local_models.append(primary_local_model)
        if local_model_list:
            for item in local_model_list.split(","):
                name = item.strip()
                if name and name not in embedding_local_models:
                    embedding_local_models.append(name)
        if not embedding_local_models:
            embedding_local_models = ["sentence-transformers/all-MiniLM-L6-v2"]
        embedding_use_local = os.getenv("EMBEDDING_USE_LOCAL", "true").lower() != "false"
        embedding_use_remote = os.getenv("EMBEDDING_USE_REMOTE", "false").lower() == "true"
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        chroma_dir = chroma_path or os.getenv("CHROMA_PATH")

        self.collection = None
        embedding_function = None
        if self.rag_enabled and self.rag_provider == "local":
            if embedding_use_local and embedding_local_models:
                for candidate in embedding_local_models:
                    try:
                        from sentence_transformers import SentenceTransformer

                        model = SentenceTransformer(candidate, token=hf_token)

                        class LocalEmbeddingFunction:
                            def name(self):
                                return f"local-st-{candidate}"

                            def _normalize(self, inp) -> List[str]:
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

                        embedding_function = LocalEmbeddingFunction()
                        self.logger.info("Using local embedding model: %s", candidate)
                        break
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning("Failed to load local embedding model %s: %s", candidate, exc)
            if embedding_function is None and embedding_use_remote and embedding_api_key:
                embedding_function = OpenAIEmbeddingFunction(
                    api_key=embedding_api_key,
                    model_name=embedding_model,
                    api_base=embedding_api_base,
                )
                self.logger.info("Using OpenAI-compatible embedding model: %s", embedding_model)

            if embedding_function:
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
                self.logger.warning(
                    "RAG enabled but no embedding function available (local disabled/unavailable and remote disabled or missing key)"
                )
                self.chroma_client = None
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

    def analyze_image(self, image_path: str, text_claim: str):
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

    def _should_stream(self, stream: Optional[bool]) -> bool:
        if stream is not None:
            return bool(stream)
        return os.getenv("LLM_STREAM", "false").lower() == "true"

    def _get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI  # type: ignore

            kwargs: Dict[str, Any] = {}
            if self.llm_api_key:
                kwargs["api_key"] = self.llm_api_key
            if self.llm_base_url:
                kwargs["base_url"] = self.llm_base_url
            self._openai_client = OpenAI(**kwargs)
            return self._openai_client
        except Exception:  # noqa: BLE001
            self._openai_client = None
            return None

    def _to_openai_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI-compatible message dicts."""
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                out.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                out.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, ToolMessage):
                out.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                payload: Dict[str, Any] = {"role": "assistant", "content": str(getattr(msg, "content", "") or "")}
                tool_calls = getattr(msg, "tool_calls", None) or []
                if tool_calls:
                    converted = []
                    for idx, call in enumerate(tool_calls):
                        call_id = call.get("id") or f"call_{idx}"
                        name = call.get("name") or ""
                        args = call.get("args") or {}
                        converted.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                            }
                        )
                    payload["tool_calls"] = converted
                out.append(payload)
            else:
                out.append({"role": "user", "content": str(getattr(msg, "content", msg))})
        return out

    def _format_openai_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        formatted: List[str] = []
        for msg in messages:
            role = msg.get("role") or "unknown"
            content = msg.get("content") or ""
            if msg.get("tool_calls"):
                content = f"tool_calls={msg.get('tool_calls')}"
            if role == "tool":
                content = f"tool_call_id={msg.get('tool_call_id')} content={content}"
            formatted.append(f"{role}: {content}")
        return formatted

    def _parse_openai_tool_calls(self, tool_calls: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse OpenAI tool_calls into:
        - internal format: [{id,name,args}]
        - raw OpenAI-ish format: [{id,type,function:{name,arguments}}]
        """
        internal: List[Dict[str, Any]] = []
        raw: List[Dict[str, Any]] = []
        if not tool_calls:
            return internal, raw

        for idx, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                tc_id = tc.get("id") or f"call_{idx}"
                tc_type = tc.get("type") or "function"
                func = tc.get("function") or {}
                name = func.get("name") or tc.get("name") or ""
                args_str = func.get("arguments") or tc.get("arguments") or ""
            else:
                tc_id = getattr(tc, "id", None) or f"call_{idx}"
                tc_type = getattr(tc, "type", None) or "function"
                func = getattr(tc, "function", None) or {}
                name = getattr(func, "name", None) or getattr(tc, "name", None) or ""
                args_str = getattr(func, "arguments", None) or getattr(tc, "arguments", None) or ""

            if args_str is None:
                args_str = ""
            if not isinstance(args_str, str):
                try:
                    args_str = json.dumps(args_str, ensure_ascii=False)
                except Exception:  # noqa: BLE001
                    args_str = str(args_str)

            args: Dict[str, Any] = {}
            if args_str:
                try:
                    args = json.loads(args_str)
                except Exception:  # noqa: BLE001
                    args = {}

            raw.append({"id": tc_id, "type": tc_type, "function": {"name": name, "arguments": args_str}})
            internal.append({"id": tc_id, "name": name, "args": args})

        return internal, raw

    def _openai_stream_once(
        self,
        messages: List[Dict[str, Any]],
        on_token: Optional[Callable[[str], None]],
        tool_choice: str = "auto",
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        One streamed ChatCompletion call.
        Returns: (content, internal_tool_calls, raw_tool_calls)
        """
        client = self._get_openai_client()
        if client is None:
            raise RuntimeError("OpenAI client is not available for streaming.")

        timeout_s = float(os.getenv("LLM_TIMEOUT", "60"))
        stream = client.chat.completions.create(
            model=self.llm_model,
            temperature=self.llm_temperature,
            messages=messages,
            tools=self.tools_schema,
            tool_choice=tool_choice,
            stream=True,
            timeout=timeout_s,
        )

        content_parts: List[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            try:
                choice = chunk.choices[0]
            except Exception:  # noqa: BLE001
                continue

            delta = getattr(choice, "delta", None) or {}
            delta_content = getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
            if delta_content:
                token = str(delta_content)
                content_parts.append(token)
                if on_token:
                    on_token(token)

            delta_tool_calls = getattr(delta, "tool_calls", None) if not isinstance(delta, dict) else delta.get("tool_calls")
            if not delta_tool_calls:
                continue

            for tc in delta_tool_calls:
                if isinstance(tc, dict):
                    idx = tc.get("index")
                    tc_id = tc.get("id")
                    tc_type = tc.get("type") or "function"
                    func = tc.get("function") or {}
                    name = func.get("name")
                    arguments = func.get("arguments")
                else:
                    idx = getattr(tc, "index", None)
                    tc_id = getattr(tc, "id", None)
                    tc_type = getattr(tc, "type", None) or "function"
                    func = getattr(tc, "function", None) or {}
                    name = getattr(func, "name", None)
                    arguments = getattr(func, "arguments", None)

                if idx is None:
                    idx = len(tool_calls_acc)
                entry = tool_calls_acc.setdefault(
                    int(idx),
                    {"id": "", "type": tc_type, "function": {"name": "", "arguments": ""}},
                )
                if tc_id:
                    entry["id"] = tc_id
                if name:
                    entry["function"]["name"] = name
                if arguments:
                    entry["function"]["arguments"] += str(arguments)

        content = "".join(content_parts)
        raw_tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())] if tool_calls_acc else []
        internal_tool_calls, raw_tool_calls = self._parse_openai_tool_calls(raw_tool_calls)
        return content, internal_tool_calls, raw_tool_calls

    def _run_openai_tool_calls(self, tool_calls: List[Dict[str, Any]], trace: List[str]) -> List[Dict[str, Any]]:
        tool_messages: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or ""
            try:
                result = self._call_tool(name, **args)
                result_str = json.dumps(result, ensure_ascii=False)
                trace.append(f"LLM tool {name}({args}) -> {result_str}")
                self.logger.info("tool_call success: %s args=%s result=%s", name, args, result_str)
            except Exception as exc:  # noqa: BLE001
                result_str = f"call {name} failed: {exc}"
                trace.append(result_str)
                self.logger.exception("tool_call failed: %s args=%s", name, args)
            tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": result_str})
        return tool_messages

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

    def chat(
        self,
        user_input: str,
        image: str | None = None,
        *,
        stream: Optional[bool] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
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
        vision_score: Optional[float] = None
        trace: List[str] = []
        conversation_log: List[Dict[str, Any]] = []

        if self.defense_enabled and self.vision_enabled and image:
            vision_checked = True
            pipeline_mode = (os.getenv("VISION_PIPELINE_MODE") or "caption_text").lower()
            trace.append(f"Vision input: text={user_input} image={image}")
            vision_consistent, vision_score, vision_caption = self.analyze_image(image, user_input)
            if vision_consistent is True:
                trace.append("Vision check: PASS")
            elif vision_consistent is False:
                trace.append("Vision check: FAIL")
            else:
                trace.append("Vision check: ERROR")
            if vision_score is not None:
                trace.append(f"Vision similarity score: {vision_score:.4f}")
            if vision_caption:
                trace.append(f"Vision caption: {vision_caption}")
            trace.append(f"Vision pipeline mode={pipeline_mode}")
            trace.append(
                "Vision models (text/mm): "
                f"{os.getenv('VISION_REMOTE_TEXT_MODEL') or 'n/a'} / "
                f"{os.getenv('VISION_REMOTE_MM_MODEL') or 'n/a'}"
            )
            # If vision check fails, intercept and return a warning without tool/RAG
            if vision_consistent is False:
                reply_text = "⚠️ 图片与描述不一致，已拦截回答。请上传匹配的图片或修改描述。"
                debug_messages = [f"{m.__class__.__name__}: {getattr(m, 'content', '')}" for m in []]
                debug_messages.append(f"AI: {reply_text}")
                self.memory.add_user_message(user_input)
                self.memory.add_ai_message(reply_text)
                return ChatResult(
                    reply=reply_text,
                    vision_checked=vision_checked,
                    vision_consistent=vision_consistent,
                    chain_context=None,
                    rag_context=None,
                    trace=trace,
                    debug_messages=debug_messages,
                    conversation_log=[],
                    trace_id=trace_id,
                )


        rag_context = ""
        if self.rag_enabled:
            rag_context = self._query_rag(user_input)
            trace.append("RAG query executed" if rag_context else "RAG query empty")
        else:
            trace.append("RAG skipped (disabled)")

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
            trace.append("Defense disabled: no auto chain snapshot/vision")

        vision_note = ""
        if vision_checked:
            if vision_consistent is True:
                status = "PASS"
            elif vision_consistent is False:
                status = "FAIL"
            else:
                status = "ERROR"
            vision_note = f"Vision consistency check: {status}"

        system_prompt = self._build_system_prompt(chain_context, rag_context, vision_note)

        history_messages = self.memory.load()
        messages: List[BaseMessage] = [SystemMessage(content=system_prompt), *history_messages, HumanMessage(content=user_input)]
        self.logger.debug("messages before LLM: %s", [m.__class__.__name__ for m in messages])
        self.logger.debug("system prompt: %s", system_prompt)

        max_rounds = int(os.getenv("TOOL_CALL_MAX_ROUNDS", "3"))
        round_idx = 1
        response: Any = None

        # Prefer OpenAI-native streaming path when enabled; fall back to LangChain invoke on errors.
        use_stream = self._should_stream(stream) and on_token is not None
        if use_stream:
            try:
                openai_messages = self._to_openai_messages(messages)
                with span("chat", {"trace_id": trace_id, "mode": self.mode, "defense": self.defense_enabled, "stream": True}):
                    while True:
                        conversation_log.append(
                            {"label": f"LLM call #{round_idx} input", "messages": self._format_openai_messages(openai_messages)}
                        )
                        with span(f"llm_call_{round_idx}", {"trace_id": trace_id, "stream": True}):
                            content, tool_calls_internal, tool_calls_raw = self._openai_stream_once(
                                openai_messages, on_token=on_token, tool_choice="auto"
                            )

                        response = {"content": content, "tool_calls": tool_calls_internal}
                        conversation_log.append(
                            {
                                "label": f"LLM call #{round_idx} output",
                                "messages": [f"assistant: {content}"],
                                "tool_calls": tool_calls_internal or None,
                            }
                        )

                        if not tool_calls_internal:
                            openai_messages.append({"role": "assistant", "content": content})
                            break

                        if round_idx >= max_rounds:
                            trace.append(f"Max tool call rounds reached ({max_rounds}); stopping.")
                            openai_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_raw})
                            break

                        openai_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls_raw})
                        with span(f"tool_exec_round_{round_idx}", {"trace_id": trace_id}):
                            tool_messages = self._run_openai_tool_calls(tool_calls_internal, trace)
                        openai_messages.extend(tool_messages)
                        round_idx += 1

                reply_text = str(response.get("content") or "")
                debug_messages = self._format_openai_messages(openai_messages)
                debug_messages.append(f"AI: {reply_text}")
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Streaming path failed; falling back to non-stream invoke: %s", exc)
                use_stream = False

        if not use_stream:
            with span("chat", {"trace_id": trace_id, "mode": self.mode, "defense": self.defense_enabled, "stream": False}):
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
                    self.logger.debug(
                        "messages after tools round %d: %s", round_idx, [m.__class__.__name__ for m in messages]
                    )
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
