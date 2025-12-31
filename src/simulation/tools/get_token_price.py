from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_token_price(token_symbol: str) -> float:
        """获取代币预言机价格（模拟 Chainlink）。"""
        return await service.get_token_price(token_symbol)
