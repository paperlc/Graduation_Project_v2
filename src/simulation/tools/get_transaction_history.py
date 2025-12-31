from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_transaction_history(address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取地址最近的 N 笔交易记录。"""
        return await service.get_transaction_history(address, limit)
