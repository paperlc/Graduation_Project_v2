from mcp.server.fastmcp import FastMCP
from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_eth_balance(address: str) -> float:
        """查询地址 ETH 余额。"""
        return await service.get_eth_balance(address)
