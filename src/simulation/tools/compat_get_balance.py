from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_balance(account: str) -> float:
        """兼容旧名：查询 ETH 余额。"""
        return await ledger.get_eth_balance(account)
