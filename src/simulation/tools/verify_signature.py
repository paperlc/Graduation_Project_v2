from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def verify_signature(message: str, signature: str, address: str) -> bool:
        """验签（示例规则：签名包含地址后6位视为通过）。"""
        return await ledger.verify_signature(message, signature, address)
