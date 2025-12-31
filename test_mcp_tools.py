"""
Quick MCP tool regression runner.

Usage:
    python scripts/test_mcp_tools.py

It copies the ledger to a temp file (so originals stay clean), sets LEDGER_FILE
to that copy, then invokes every registered tool with sample arguments.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.mcp_client.client import MCPToolClient  # noqa: E402
DEFAULT_LEDGER = ROOT / "data" / "ledger" / "ledger.json"

# (tool_name, kwargs)
TEST_CASES: List[Tuple[str, Dict[str, Any]]] = [
    ("get_eth_balance", {"address": "alice"}),
    ("get_token_balance", {"address": "alice", "token_symbol": "USDT"}),
    ("get_transaction_history", {"address": "alice", "limit": 2}),
    ("get_contract_bytecode", {"address": "0xrouter0000000000000000000000000000000000"}),
    ("resolve_ens_domain", {"domain_name": "vitalik.eth"}),
    ("get_token_price", {"token_symbol": "ETH"}),
    ("check_address_reputation", {"address": "0xdeadbeef000000000000000000000000000000"}),
    ("simulate_transaction", {"to": "malicious_router", "value": 1.5, "data": "0xabcd"}),
    ("verify_contract_owner", {"contract_address": "0xrouter0000000000000000000000000000000000"}),
    ("check_token_approval", {"owner": "alice", "spender": "dex_router"}),
    ("verify_signature", {"message": "pay alice", "signature": "sig-for-000abcde", "address": "0xdeadbeef000000000000000000000000000abcde"}),
    ("transfer_eth", {"to_address": "bob", "amount": 10.0, "sender": "alice"}),
    ("swap_tokens", {"token_in": "USDT", "token_out": "UNI", "amount": 10.0, "address": "alice"}),
    ("approve_token", {"spender": "new_dapp", "amount": 50.0, "owner": "alice"}),
    ("revoke_approval", {"spender": "dex_router", "owner": "alice"}),
    ("get_liquidity_pool_info", {"token_address": "USDT-ETH"}),
    ("bridge_asset", {"token": "USDT", "target_chain": "arbitrum"}),
    ("stake_tokens", {"protocol": "lending_pool", "amount": 25.0}),
    ("get_balance", {"account": "alice"}),  # compat
    ("transfer", {"sender": "alice", "recipient": "bob", "amount": 1.0}),  # compat
]


def copy_ledger() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="ledger_test_"))
    dest = temp_dir / "ledger.json"
    shutil.copy(DEFAULT_LEDGER, dest)
    return dest


def run_case(client: MCPToolClient, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = client.call_tool(name, **params)
        return {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> None:
    ledger_copy = copy_ledger()
    os.environ["LEDGER_FILE"] = str(ledger_copy)
    os.environ.setdefault("MCP_SERVER_CMD", "python -m src.simulation.server")

    print(f"[info] Using temp ledger: {ledger_copy}")
    print(f"[info] MCP_SERVER_CMD={os.environ['MCP_SERVER_CMD']}")

    client = MCPToolClient(server_cmd=os.environ["MCP_SERVER_CMD"], server_url=os.getenv("MCP_SERVER_URL"))

    results = []
    for name, params in TEST_CASES:
        res = run_case(client, name, params)
        results.append((name, res))

    print("\n=== MCP tool regression results ===")
    for name, res in results:
        status = "OK " if res["ok"] else "FAIL"
        payload = res.get("result") if res["ok"] else res.get("error")
        pretty = json.dumps(payload, ensure_ascii=False) if isinstance(payload, (dict, list)) else str(payload)
        print(f"{status:4} {name}: {pretty}")


if __name__ == "__main__":
    main()
