from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def transfer_eth(to_address: str, amount: float, sender: str | None = None) -> Dict[str, Any]:
        """发送 ETH（写入文本账本）。"""
        return await service.transfer_eth(to_address, amount, from_address=sender)
