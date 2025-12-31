from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def check_address_reputation(address: str) -> str:
        """查询地址声誉/黑名单状态。"""
        return await service.check_address_reputation(address)
