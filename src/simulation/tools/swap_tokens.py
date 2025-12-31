from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def swap_tokens(token_in: str, token_out: str, amount: float, address: str | None = None) -> Dict[str, Any]:
        """模拟 DEX 兑换。"""
        return await service.swap_tokens(token_in, token_out, amount, address)
