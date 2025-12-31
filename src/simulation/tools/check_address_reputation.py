from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def check_address_reputation(address: str) -> str:
        """查询地址声誉/黑名单状态。"""
        return await ledger.check_address_reputation(address)
