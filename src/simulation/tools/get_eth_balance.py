from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_eth_balance(address: str) -> float:
        """查询地址 ETH 余额。"""
        return await ledger.get_eth_balance(address)
