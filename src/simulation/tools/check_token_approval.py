from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def check_token_approval(owner: str, spender: str) -> float:
        """查询授权额度。"""
        return await service.check_token_approval(owner, spender)
