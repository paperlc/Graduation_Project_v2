"""Text-backed ledger with helper methods for simulated chain operations."""

from __future__ import annotations

import asyncio
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[2] / "data" / "ledger" / "ledger.json"
)
logger = logging.getLogger(__name__)


def _ensure_ledger_file(path: Path) -> None:
    """Raise a readable error if the ledger file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Ledger file not found at {path}. Copy data/ledger/ledger.json as a template and retry."
        )


class Ledger:
    """Text-backed ledger with simple simulation helpers."""

    def __init__(self, ledger_path: Path | str | None = None):
        env_path = os.getenv("LEDGER_FILE")
        self.ledger_path = Path(ledger_path or env_path or _DEFAULT_LEDGER)
        _ensure_ledger_file(self.ledger_path)
        self._lock = asyncio.Lock()

    async def _load(self) -> Dict[str, Any]:
        with self.ledger_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    async def _save(self, data: Dict[str, Any]) -> None:
        text = json.dumps(data, ensure_ascii=False, indent=2)
        self.ledger_path.write_text(text, encoding="utf-8")

    # ---------- 读操作 ----------
    async def get_eth_balance(self, address: str) -> float:
        logger.debug("get_eth_balance address=%s", address)
        async with self._lock:
            data = await self._load()
            return float(data.get("accounts", {}).get(address, 0.0))

    async def get_token_balance(self, address: str, token_symbol: str) -> float:
        logger.debug("get_token_balance address=%s token=%s", address, token_symbol)
        async with self._lock:
            data = await self._load()
            balances = data.get("token_balances", {}).get(address, {})
            return float(balances.get(token_symbol.upper(), 0.0))

    async def get_transaction_history(self, address: str, limit: int = 5) -> List[Dict[str, Any]]:
        logger.debug("get_transaction_history address=%s limit=%s", address, limit)
        async with self._lock:
            data = await self._load()
            txs = data.get("transactions", [])
            related = [tx for tx in txs if tx.get("from") == address or tx.get("to") == address]
            return related[-limit:]

    async def get_contract_bytecode(self, address: str) -> str:
        logger.debug("get_contract_bytecode address=%s", address)
        async with self._lock:
            data = await self._load()
            return data.get("contracts", {}).get(address, {}).get("bytecode", "")

    async def resolve_ens_domain(self, domain_name: str) -> str:
        logger.debug("resolve_ens_domain domain=%s", domain_name)
        async with self._lock:
            data = await self._load()
            return data.get("ens", {}).get(domain_name, "")

    async def get_token_price(self, token_symbol: str) -> float:
        logger.debug("get_token_price token=%s", token_symbol)
        async with self._lock:
            data = await self._load()
            return float(data.get("prices", {}).get(token_symbol.upper(), 0.0))

    async def check_address_reputation(self, address: str) -> str:
        logger.debug("check_address_reputation address=%s", address)
        async with self._lock:
            data = await self._load()
            rep = data.get("reputations", {}).get(address.lower())
            return rep or "unknown"

    async def verify_contract_owner(self, contract_address: str) -> str:
        logger.debug("verify_contract_owner contract=%s", contract_address)
        async with self._lock:
            data = await self._load()
            return data.get("contracts", {}).get(contract_address, {}).get("owner", "")

    async def check_token_approval(self, owner: str, spender: str) -> float:
        logger.debug("check_token_approval owner=%s spender=%s", owner, spender)
        async with self._lock:
            data = await self._load()
            approvals = data.get("approvals", {}).get(owner, {})
            return float(approvals.get(spender, 0.0))

    async def get_liquidity_pool_info(self, token_address: str) -> Dict[str, Any]:
        logger.debug("get_liquidity_pool_info token=%s", token_address)
        async with self._lock:
            data = await self._load()
            return data.get("liquidity_pools", {}).get(token_address, {})

    # ---------- 写/模拟操作 ----------
    async def _record_tx(
        self, data: Dict[str, Any], tx_record: Dict[str, Any], accounts: Dict[str, float] | None = None
    ) -> None:
        data.setdefault("transactions", []).append(tx_record)
        if accounts is not None:
            data["accounts"] = accounts
        await self._save(data)

    async def transfer_eth(self, to_address: str, amount: float, from_address: str | None = None) -> Dict[str, Any]:
        logger.info("transfer_eth from=%s to=%s amount=%s", from_address, to_address, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")

        async with self._lock:
            data = await self._load()
            accounts = data.setdefault("accounts", {})
            sender = from_address or data.get("meta", {}).get("default_actor", "treasury")
            sender_balance = accounts.get(sender, 0.0)
            if sender_balance < amount:
                raise ValueError("Insufficient funds.")

            accounts[sender] = sender_balance - amount
            accounts[to_address] = accounts.get(to_address, 0.0) + amount

            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            tx_record = {
                "tx": f"eth-{now}",
                "from": sender,
                "to": to_address,
                "amount": amount,
                "token": "ETH",
                "timestamp": now,
            }
            await self._record_tx(data, tx_record, accounts)
            return {"from": sender, "to": to_address, "amount": amount, "sender_balance": accounts[sender]}

    async def swap_tokens(
        self, token_in: str, token_out: str, amount: float, address: str | None = None
    ) -> Dict[str, Any]:
        logger.info("swap_tokens actor=%s in=%s out=%s amount=%s", address, token_in, token_out, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")

        async with self._lock:
            data = await self._load()
            actor = address or data.get("meta", {}).get("default_actor", "treasury")
            balances = data.setdefault("token_balances", {}).setdefault(actor, {})
            prices = data.get("prices", {})

            token_in_u = token_in.upper()
            token_out_u = token_out.upper()
            in_balance = balances.get(token_in_u, 0.0)
            if in_balance < amount:
                raise ValueError("Insufficient token balance.")

            price_in = prices.get(token_in_u, 1.0)
            price_out = prices.get(token_out_u, 1.0)
            amount_out = round(amount * price_in / max(price_out, 1e-8), 6)

            balances[token_in_u] = in_balance - amount
            balances[token_out_u] = balances.get(token_out_u, 0.0) + amount_out

            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            tx_record = {
                "tx": f"swap-{now}",
                "from": actor,
                "to": actor,
                "amount": amount,
                "token": token_in_u,
                "amount_out": amount_out,
                "token_out": token_out_u,
                "timestamp": now,
            }
            await self._record_tx(data, tx_record, data.get("accounts"))
            return {"token_in": token_in_u, "token_out": token_out_u, "amount_out": amount_out, "actor": actor}

    async def approve_token(self, owner: str, spender: str, amount: float) -> Dict[str, Any]:
        logger.info("approve_token owner=%s spender=%s amount=%s", owner, spender, amount)
        if amount < 0:
            raise ValueError("Amount must be non-negative.")

        async with self._lock:
            data = await self._load()
            approvals = data.setdefault("approvals", {}).setdefault(owner, {})
            approvals[spender] = amount
            await self._save(data)
            return {"owner": owner, "spender": spender, "amount": amount}

    async def revoke_approval(self, owner: str, spender: str) -> Dict[str, Any]:
        logger.info("revoke_approval owner=%s spender=%s", owner, spender)
        async with self._lock:
            data = await self._load()
            approvals = data.get("approvals", {}).get(owner, {})
            if spender in approvals:
                approvals[spender] = 0.0
            await self._save(data)
            return {"owner": owner, "spender": spender, "amount": 0.0}

    async def simulate_transaction(self, to: str, value: float, data_field: str | None = None) -> Dict[str, Any]:
        """交易预执行，简单规则：检查余额与黑名单。"""
        logger.info("simulate_transaction to=%s value=%s", to, value)
        async with self._lock:
            data = await self._load()
            accounts = data.get("accounts", {})
            sender = data.get("meta", {}).get("default_actor", "treasury")
            sender_balance = accounts.get(sender, 0.0)
            reputation = data.get("reputations", {}).get(to.lower())
            return {
                "sender": sender,
                "can_afford": sender_balance >= value,
                "target_reputation": reputation or "unknown",
                "estimated_fee": 0.001,
                "note": "blacklist" if reputation else "ok",
                "data": data_field or "",
            }

    async def verify_signature(self, message: str, signature: str, address: str) -> bool:
        """
        简化版验签：仅做格式/占位校验。
        规则：签名包含地址后 6 位视为通过。
        """
        logger.info("verify_signature address=%s", address)
        suffix = address.lower()[-6:]
        return suffix in signature.lower()

    async def bridge_asset(self, token: str, target_chain: str) -> Dict[str, Any]:
        logger.info("bridge_asset token=%s target=%s", token, target_chain)
        async with self._lock:
            data = await self._load()
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            record = {"token": token.upper(), "to_chain": target_chain, "timestamp": now}
            data.setdefault("bridges", []).append(record)
            await self._save(data)
            return record

    async def stake_tokens(self, protocol: str, amount: float) -> Dict[str, Any]:
        logger.info("stake_tokens protocol=%s amount=%s", protocol, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        async with self._lock:
            data = await self._load()
            actor = data.get("meta", {}).get("default_actor", "treasury")
            stakes = data.setdefault("stakes", {}).setdefault(actor, [])
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            stake_record = {"protocol": protocol, "amount": amount, "timestamp": now}
            stakes.append(stake_record)
            await self._save(data)
            return {"actor": actor, "protocol": protocol, "amount": amount}
