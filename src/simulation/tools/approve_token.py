from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def approve_token(spender: str, amount: float, owner: str | None = None) -> Dict[str, Any]:
        """授权代币额度。"""
        owner_addr = owner or "treasury"
        return await ledger.approve_token(owner_addr, spender, amount)
