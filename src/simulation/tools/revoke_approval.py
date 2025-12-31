from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def revoke_approval(spender: str, owner: str | None = None) -> Dict[str, Any]:
        """撤销授权。"""
        owner_addr = owner or "treasury"
        return await service.revoke_approval(owner_addr, spender)
