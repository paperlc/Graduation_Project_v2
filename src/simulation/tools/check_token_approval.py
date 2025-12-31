from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def check_token_approval(owner: str, spender: str) -> float:
        """查询授权额度。"""
        return await ledger.check_token_approval(owner, spender)
