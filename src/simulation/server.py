"""MCP server wiring: load text-backed ledger and register tool modules."""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import import_module
from typing import List

import logging
from mcp.server.fastmcp import FastMCP

from .ledger import Ledger
from .services.types import LedgerService

# Tool modules (one file per tool for easy add/remove)
TOOL_MODULES: List[str] = [
    "src.simulation.tools.get_eth_balance",
    "src.simulation.tools.get_token_balance",
    "src.simulation.tools.get_transaction_history",
    "src.simulation.tools.get_contract_bytecode",
    "src.simulation.tools.resolve_ens_domain",
    "src.simulation.tools.get_token_price",
    "src.simulation.tools.check_address_reputation",
    "src.simulation.tools.simulate_transaction",
    "src.simulation.tools.verify_contract_owner",
    "src.simulation.tools.check_token_approval",
    "src.simulation.tools.verify_signature",
    "src.simulation.tools.transfer_eth",
    "src.simulation.tools.swap_tokens",
    "src.simulation.tools.approve_token",
    "src.simulation.tools.revoke_approval",
    "src.simulation.tools.get_liquidity_pool_info",
    "src.simulation.tools.bridge_asset",
    "src.simulation.tools.stake_tokens",
    "src.simulation.tools.compat_get_balance",
    "src.simulation.tools.compat_transfer",
]


logger = logging.getLogger(__name__)


def start_health_server(host: str, port: int, ready_flag: dict) -> threading.Thread:
    """Start a lightweight HTTP server for health/readiness probes."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: D401
            # Silence default stdout logging
            return

        def do_GET(self):  # noqa: N802
            if self.path in ("/healthz", "/livez"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
                return
            if self.path in ("/readyz", "/ready"):
                status = 200 if ready_flag.get("ready") else 503
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                payload = b'{"status":"ready"}' if status == 200 else b'{"status":"not-ready"}'
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

    httpd = HTTPServer((host, port), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    logger.info("Health server started on http://%s:%s (healthz/readyz)", host, port)
    ready_flag["server"] = httpd
    return thread


def build_server(host: str | None = None, port: int | None = None, sse_path: str | None = None) -> FastMCP:
    logger.info("Building MCP server with ledger and %d tools", len(TOOL_MODULES))
    mcp = FastMCP(
        "web3-ledger",
        host=host or "127.0.0.1",
        port=port or 8000,
        sse_path=sse_path or "/sse",
    )
    service: LedgerService = Ledger()

    for module_path in TOOL_MODULES:
        module = import_module(module_path)
        if hasattr(module, "register"):
            module.register(mcp, service)
            logger.info("Registered tool module: %s", module_path)
        else:
            logger.warning("Module %s missing register()", module_path)

    return mcp


def run():
    """Entry point for the MCP simulation server."""
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8001"))
    sse_path = os.getenv("MCP_SSE_PATH", "/sse")
    health_host = os.getenv("HEALTH_HOST", "0.0.0.0")
    health_port = int(os.getenv("HEALTH_PORT", "8081"))
    ready_flag: dict = {"ready": False}

    health_thread = start_health_server(health_host, health_port, ready_flag)

    mcp = build_server(host=host, port=port, sse_path=sse_path)
    logger.info("Starting MCP server... transport=%s host=%s port=%s sse_path=%s", transport, host, port, sse_path)
    ready_flag["ready"] = True

    if transport == "stdio":
        mcp.run()
    else:
        # fastmcp supports "http", "sse", "streamable-http"
        mcp.run(transport=transport)


if __name__ == "__main__":
    run()
