from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def simulate_transaction(to: str, value: float, data: str | None = None) -> Dict[str, Any]:
        """交易预执行，检查余额与黑名单。"""
        return await service.simulate_transaction(to, value, data)
