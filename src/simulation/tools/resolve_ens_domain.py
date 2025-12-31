from mcp.server.fastmcp import FastMCP

from src.simulation.ledger import Ledger


def register(mcp: FastMCP, ledger: Ledger) -> None:
    @mcp.tool()
    async def resolve_ens_domain(domain_name: str) -> str:
        """解析 ENS 域名为链上地址。"""
        return await ledger.resolve_ens_domain(domain_name)
