from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_liquidity_pool_info(token_address: str) -> Dict[str, Any]:
        """查询流动性池信息。"""
        return await service.get_liquidity_pool_info(token_address)
