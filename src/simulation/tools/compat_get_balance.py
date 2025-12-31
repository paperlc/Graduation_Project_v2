from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_balance(account: str) -> float:
        """兼容旧名：查询 ETH 余额。"""
        return await service.get_eth_balance(account)
