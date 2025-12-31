"""MCP client helper to connect to the FastMCP server and call tools."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from typing import Any, Dict, Optional

try:
    # mcp >= 1.25 provides stdio_client + StdioServerParameters
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client
    from src.utils.telemetry import span, get_trace_id
except Exception:  # pragma: no cover
    stdio_client = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    ClientSession = None  # type: ignore
    sse_client = None  # type: ignore

logger = logging.getLogger(__name__)


class MCPToolClient:
    """Thin wrapper around MCP client session for tool calls with timeout/retry."""

    def __init__(
        self,
        server_cmd: Optional[str] = None,
        server_url: Optional[str] = None,
        server_headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 15.0,
        retries: int = 1,
    ):
        self.server_cmd = server_cmd
        self.server_url = server_url
        self.server_headers = server_headers or {}
        self.timeout_seconds = timeout_seconds
        self.retries = max(retries, 1)
        self._tools: Dict[str, Dict[str, Any]] = {}

    async def call_tool_async(self, name: str, **kwargs) -> Any:
        """
        Create a fresh stdio session per call (avoids missing event loop issues).
        Every error is logged and raised so the frontend can surface it.
        """
        if not self.server_cmd and not self.server_url:
            raise RuntimeError("MCP_SERVER_CMD or MCP_SERVER_URL is not set; cannot call tools.")
        if ClientSession is None:
            raise ImportError("mcp.client is unavailable; check mcp version.")

        if self.server_url:
            if sse_client is None:
                raise ImportError("mcp.client.sse unavailable; check mcp version.")
            return await self._call_via_sse(name, **kwargs)

        if stdio_client is None or StdioServerParameters is None:
            raise ImportError("mcp.client.stdio.stdio_client unavailable; check mcp version.")
        cmd_parts = shlex.split(self.server_cmd)
        if not cmd_parts:
            raise ValueError("MCP_SERVER_CMD is empty.")
        command = cmd_parts[0]
        args = cmd_parts[1:]
        server_cfg = StdioServerParameters(command=command, args=args, env=os.environ.copy(), cwd=os.getcwd())

        logger.info("MCP stdio connect, cmd=%s args=%s", command, args, extra={"trace_id": get_trace_id()})
        async with stdio_client(server_cfg) as (read_stream, write_stream):
            # ClientSession must run as a context manager to start the receive loop.
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                resp_tools = await session.list_tools()
                tools = {tool.name: tool for tool in resp_tools.tools}
                logger.debug("MCP available tools: %s", list(tools.keys()))
                if name not in tools:
                    raise ValueError(f"Tool {name} not found in MCP registry.")

                attempt = 0
                last_err = None
                while attempt < self.retries:
                    attempt += 1
                    try:
                        with span(f"tool_call:{name}", {"trace_id": get_trace_id(), "transport": "stdio", "attempt": attempt}):
                            logger.info(
                                "MCP call_tool start: %s args=%s attempt=%s", name, kwargs, attempt, extra={"trace_id": get_trace_id()}
                            )
                            resp = await asyncio.wait_for(session.call_tool(name, kwargs), timeout=self.timeout_seconds)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        logger.warning("MCP call_tool error attempt=%s name=%s err=%s", attempt, name, exc, extra={"trace_id": get_trace_id()})
                        if attempt >= self.retries:
                            raise
                outputs = []
                for item in resp.content:
                    if hasattr(item, "data"):
                        payload = item.data
                    elif hasattr(item, "text"):
                        payload = item.text
                    elif hasattr(item, "content"):
                        payload = item.content
                    else:
                        payload = item.model_dump()  # fallback for unknown content types
                    outputs.append({"type": getattr(item, "type", type(item).__name__), "data": payload})
                logger.info("MCP call_tool done: %s outputs=%s", name, outputs, extra={"trace_id": get_trace_id()})
                if len(outputs) == 1:
                    return outputs[0]["data"]
                return outputs

    async def _call_via_sse(self, name: str, **kwargs) -> Any:
        logger.info("MCP SSE connect, url=%s", self.server_url, extra={"trace_id": get_trace_id()})
        async with sse_client(self.server_url, headers=self.server_headers) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                resp_tools = await session.list_tools()
                tools = {tool.name: tool for tool in resp_tools.tools}
                logger.debug("MCP available tools: %s", list(tools.keys()))
                if name not in tools:
                    raise ValueError(f"Tool {name} not found in MCP registry.")

                attempt = 0
                while attempt < self.retries:
                    attempt += 1
                    try:
                        with span(f"tool_call:{name}", {"trace_id": get_trace_id(), "transport": "sse", "attempt": attempt}):
                            logger.info(
                                "MCP call_tool start: %s args=%s attempt=%s", name, kwargs, attempt, extra={"trace_id": get_trace_id()}
                            )
                            resp = await asyncio.wait_for(session.call_tool(name, kwargs), timeout=self.timeout_seconds)
                        break
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("MCP call_tool error attempt=%s name=%s err=%s", attempt, name, exc, extra={"trace_id": get_trace_id()})
                        if attempt >= self.retries:
                            raise
                outputs = []
                for item in resp.content:
                    if hasattr(item, "data"):
                        payload = item.data
                    elif hasattr(item, "text"):
                        payload = item.text
                    elif hasattr(item, "content"):
                        payload = item.content
                    else:
                        payload = item.model_dump()
                    outputs.append({"type": getattr(item, "type", type(item).__name__), "data": payload})
                logger.info("MCP call_tool done: %s outputs=%s", name, outputs, extra={"trace_id": get_trace_id()})
                if len(outputs) == 1:
                    return outputs[0]["data"]
                return outputs

    def call_tool(self, name: str, **kwargs) -> Any:
        """
        Synchronous wrapper; uses asyncio.run when no running loop is present.
        We intentionally do not silence RuntimeError to keep stack traces loud.
        """
        try:
            logger.debug("Using asyncio.run for tool %s", name)
            return asyncio.run(self.call_tool_async(name, **kwargs))
        except RuntimeError:
            # Raised when an event loop is already running; surface loudly.
            logger.exception("call_tool cannot run inside an active event loop; use call_tool_async instead.")
            raise


def make_mcp_tool_caller() -> MCPToolClient:
    server_cmd = os.getenv("MCP_SERVER_CMD")
    server_url = os.getenv("MCP_SERVER_URL")
    timeout = float(os.getenv("TOOL_CALL_TIMEOUT", "15"))
    retries = int(os.getenv("TOOL_CALL_RETRIES", "1"))
    headers_env = os.getenv("MCP_SERVER_HEADERS")
    server_headers: Dict[str, str] = {}
    if headers_env:
        try:
            import json

            server_headers = json.loads(headers_env)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse MCP_SERVER_HEADERS: %s", exc)
    return MCPToolClient(
        server_cmd=server_cmd,
        server_url=server_url,
        server_headers=server_headers,
        timeout_seconds=timeout,
        retries=retries,
    )
