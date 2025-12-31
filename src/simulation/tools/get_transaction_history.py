from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_transaction_history(address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取地址最近的 N 笔交易记录。"""
        return await ledger.get_transaction_history(address, limit)
