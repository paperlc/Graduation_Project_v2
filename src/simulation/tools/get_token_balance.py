from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_token_balance(address: str, token_symbol: str) -> float:
        """查询地址的 ERC-20 代币余额。"""
        return await ledger.get_token_balance(address, token_symbol)
