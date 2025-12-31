from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def verify_signature(message: str, signature: str, address: str) -> bool:
        """验签（示例规则：签名包含地址后6位视为通过）。"""
        return await service.verify_signature(message, signature, address)
