from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def swap_tokens(token_in: str, token_out: str, amount: float, address: str | None = None) -> Dict[str, Any]:
        """模拟 DEX 兑换。"""
        return await ledger.swap_tokens(token_in, token_out, amount, address)
