from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def get_contract_bytecode(address: str) -> str:
        """获取地址的合约字节码（判断是否合约）。"""
        return await ledger.get_contract_bytecode(address)
