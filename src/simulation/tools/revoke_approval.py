from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def revoke_approval(spender: str, owner: str | None = None) -> Dict[str, Any]:
        """撤销授权。"""
        owner_addr = owner or "treasury"
        return await ledger.revoke_approval(owner_addr, spender)
