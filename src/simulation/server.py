"""MCP server wiring: load text-backed ledger and register tool modules."""

from __future__ import annotations

from importlib import import_module
from typing import List

from mcp.server.fastmcp import FastMCP

from .ledger import Ledger

# 新增工具模块列表（单文件单工具，方便增删）
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


def build_server() -> FastMCP:
    mcp = FastMCP("web3-ledger")
    ledger = Ledger()

    for module_path in TOOL_MODULES:
        module = import_module(module_path)
        if hasattr(module, "register"):
            module.register(mcp, ledger)

    return mcp


def run():
    """Entry point for the MCP simulation server."""
    mcp = build_server()
    mcp.run()


if __name__ == "__main__":
    run()
