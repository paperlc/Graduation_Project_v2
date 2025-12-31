from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def stake_tokens(protocol: str, amount: float) -> Dict[str, Any]:
        """模拟质押。"""
        return await service.stake_tokens(protocol, amount)
