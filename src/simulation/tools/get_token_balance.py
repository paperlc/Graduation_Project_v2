from mcp.server.fastmcp import FastMCP
from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_token_balance(address: str, token_symbol: str) -> float:
        """查询地址的 ERC-20 代币余额。"""
        return await service.get_token_balance(address, token_symbol)
