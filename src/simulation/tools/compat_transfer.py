from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def transfer(sender: str, recipient: str, amount: float) -> Dict[str, Any]:
        """兼容旧名：ETH 转账。"""
        return await service.transfer_eth(recipient, amount, from_address=sender)
