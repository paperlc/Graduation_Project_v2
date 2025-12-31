from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def simulate_transaction(to: str, value: float, data: str | None = None) -> Dict[str, Any]:
        """交易预执行，检查余额与黑名单。"""
        return await ledger.simulate_transaction(to, value, data)
