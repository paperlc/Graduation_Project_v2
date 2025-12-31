from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def get_contract_bytecode(address: str) -> str:
        """获取地址的合约字节码（判断是否合约）。"""
        return await service.get_contract_bytecode(address)
