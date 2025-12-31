from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def bridge_asset(token: str, target_chain: str) -> Dict[str, Any]:
        """模拟跨链。"""
        return await ledger.bridge_asset(token, target_chain)
