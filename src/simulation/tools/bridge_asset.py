from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def bridge_asset(token: str, target_chain: str) -> Dict[str, Any]:
        """模拟跨链。"""
        return await service.bridge_asset(token, target_chain)
