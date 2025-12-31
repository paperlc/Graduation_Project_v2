from mcp.server.fastmcp import FastMCP

from src.simulation.services.types import LedgerService


def register(mcp: FastMCP, service: LedgerService) -> None:
    @mcp.tool()
    async def resolve_ens_domain(domain_name: str) -> str:
        """解析 ENS 域名为链上地址。"""
        return await service.resolve_ens_domain(domain_name)
