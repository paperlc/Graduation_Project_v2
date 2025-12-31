from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_token_price(token_symbol: str) -> float:
        """获取代币预言机价格（模拟 Chainlink）。"""
        return await ledger.get_token_price(token_symbol)
