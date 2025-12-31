from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def verify_contract_owner(contract_address: str) -> str:
        """查询合约 owner。"""
        return await service.verify_contract_owner(contract_address)
