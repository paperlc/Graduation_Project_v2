from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def verify_contract_owner(contract_address: str) -> str:
        """查询合约 owner。"""
        return await ledger.verify_contract_owner(contract_address)
