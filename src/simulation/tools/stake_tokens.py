from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def stake_tokens(protocol: str, amount: float) -> Dict[str, Any]:
        """模拟质押。"""
        return await ledger.stake_tokens(protocol, amount)
