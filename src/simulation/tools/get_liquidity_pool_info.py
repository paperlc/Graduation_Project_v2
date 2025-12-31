from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_liquidity_pool_info(token_address: str) -> Dict[str, Any]:
        """查询流动性池信息。"""
        return await ledger.get_liquidity_pool_info(token_address)
