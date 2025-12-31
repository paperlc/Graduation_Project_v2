from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def transfer(sender: str, recipient: str, amount: float) -> Dict[str, Any]:
        """兼容旧名：ETH 转账。"""
        return await ledger.transfer_eth(recipient, amount, from_address=sender)
