"""SQLite-backed ledger with helper methods for simulated chain operations."""

from __future__ import annotations

import asyncio
import json
import os
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import sqlite3

from .db import connect, load_initial_state

_DEFAULT_JSON = Path(__file__).resolve().parents[2] / "data" / "ledger" / "ledger.json"
_DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "ledger" / "ledger.db"
_SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "ledger" / "snapshots"
logger = logging.getLogger(__name__)


class Ledger:
    """SQLite-backed ledger with simple simulation helpers."""

    def __init__(self, db_path: Path | str | None = None, json_seed: Path | str | None = None):
        env_db = os.getenv("LEDGER_DB")
        self.db_path = Path(db_path or env_db or _DEFAULT_DB)
        self.json_seed = Path(json_seed or os.getenv("LEDGER_FILE") or _DEFAULT_JSON)
        self._lock = asyncio.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def _init_db(self) -> None:
        # Ensure directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) AS cnt FROM accounts")
        row = cur.fetchone()
        if row and row["cnt"] > 0:
            conn.close()
            return
        if not self.json_seed.exists():
            conn.close()
            raise FileNotFoundError(f"Ledger seed file not found at {self.json_seed}")
        data = json.loads(self.json_seed.read_text(encoding="utf-8"))
        load_initial_state(
            conn,
            accounts=data.get("accounts", {}),
            token_balances=data.get("token_balances", {}),
            approvals=data.get("approvals", {}),
            contracts=data.get("contracts", {}),
            ens=data.get("ens", {}),
            prices=data.get("prices", {}),
            reputations=data.get("reputations", {}),
            transactions=data.get("transactions", []),
            liquidity_pools=data.get("liquidity_pools", {}),
        )
        conn.close()

    def _snapshot(self) -> None:
        ts = datetime.utcnow().isoformat(timespec="seconds").replace(":", "-")
        target = _SNAPSHOT_DIR / f"ledger-{ts}.db"
        shutil.copy(self.db_path, target)
        logger.info("Snapshot saved to %s", target)

    def _audit(self, conn: sqlite3.Connection, action: str, payload: Dict[str, Any]) -> None:
        conn.execute("INSERT INTO audit(action, payload) VALUES(?, ?)", (action, json.dumps(payload, ensure_ascii=False)))

    def _idempotent_result(self, conn: sqlite3.Connection, key: Optional[str]) -> Optional[Any]:
        if not key:
            return None
        cur = conn.execute("SELECT result FROM idempotency WHERE key = ?", (key,))
        row = cur.fetchone()
        if row:
            return json.loads(row["result"])
        return None

    def _store_idempotent(self, conn: sqlite3.Connection, key: Optional[str], result: Any) -> None:
        if not key:
            return
        conn.execute(
            "INSERT OR REPLACE INTO idempotency(key, result) VALUES(?, ?)",
            (key, json.dumps(result, ensure_ascii=False)),
        )

    # ---------- 读操作 ----------
    async def get_eth_balance(self, address: str) -> float:
        logger.debug("get_eth_balance address=%s", address)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT balance FROM accounts WHERE address = ?", (address,))
            row = cur.fetchone()
            conn.close()
            return float(row["balance"]) if row else 0.0

    async def get_token_balance(self, address: str, token_symbol: str) -> float:
        logger.debug("get_token_balance address=%s token=%s", address, token_symbol)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "SELECT balance FROM token_balances WHERE address = ? AND token = ?", (address, token_symbol.upper())
            )
            row = cur.fetchone()
            conn.close()
            return float(row["balance"]) if row else 0.0

    async def get_transaction_history(self, address: str, limit: int = 5) -> List[Dict[str, Any]]:
        logger.debug("get_transaction_history address=%s limit=%s", address, limit)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "SELECT tx_id as tx, from_addr as 'from', to_addr as 'to', amount, token, timestamp, memo "
                "FROM transactions WHERE from_addr = ? OR to_addr = ? ORDER BY timestamp DESC LIMIT ?",
                (address, address, limit),
            )
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows

    async def get_contract_bytecode(self, address: str) -> str:
        logger.debug("get_contract_bytecode address=%s", address)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT bytecode FROM contracts WHERE address = ?", (address,))
            row = cur.fetchone()
            conn.close()
            return row["bytecode"] if row else ""

    async def resolve_ens_domain(self, domain_name: str) -> str:
        logger.debug("resolve_ens_domain domain=%s", domain_name)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT address FROM ens WHERE domain = ?", (domain_name,))
            row = cur.fetchone()
            conn.close()
            return row["address"] if row else ""

    async def get_token_price(self, token_symbol: str) -> float:
        logger.debug("get_token_price token=%s", token_symbol)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT price FROM prices WHERE token = ?", (token_symbol.upper(),))
            row = cur.fetchone()
            conn.close()
            return float(row["price"]) if row else 0.0

    async def check_address_reputation(self, address: str) -> str:
        logger.debug("check_address_reputation address=%s", address)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT status FROM reputations WHERE address = ?", (address.lower(),))
            row = cur.fetchone()
            conn.close()
            return row["status"] if row else "unknown"

    async def verify_contract_owner(self, contract_address: str) -> str:
        logger.debug("verify_contract_owner contract=%s", contract_address)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT owner FROM contracts WHERE address = ?", (contract_address,))
            row = cur.fetchone()
            conn.close()
            return row["owner"] if row else ""

    async def check_token_approval(self, owner: str, spender: str) -> float:
        logger.debug("check_token_approval owner=%s spender=%s", owner, spender)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute("SELECT amount FROM approvals WHERE owner = ? AND spender = ?", (owner, spender))
            row = cur.fetchone()
            conn.close()
            return float(row["amount"]) if row else 0.0

    async def get_liquidity_pool_info(self, token_address: str) -> Dict[str, Any]:
        logger.debug("get_liquidity_pool_info token=%s", token_address)
        async with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "SELECT pool, token0, token1, liquidity_usd, fee_bps FROM liquidity_pools WHERE pool = ?",
                (token_address,),
            )
            row = cur.fetchone()
            conn.close()
            return dict(row) if row else {}

    # ---------- 写/模拟操作 ----------
    async def transfer_eth(
        self, to_address: str, amount: float, from_address: str | None = None, idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        logger.info("transfer_eth from=%s to=%s amount=%s", from_address, to_address, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            cur = conn.cursor()
            sender = from_address or self._default_actor(conn)
            cur.execute("SELECT balance FROM accounts WHERE address = ?", (sender,))
            row = cur.fetchone()
            sender_balance = float(row["balance"]) if row else 0.0
            if sender_balance < amount:
                conn.close()
                raise ValueError("Insufficient funds.")
            cur.execute("UPDATE accounts SET balance = balance - ? WHERE address = ?", (amount, sender))
            cur.execute(
                "INSERT INTO accounts(address, balance) VALUES(?, ?) ON CONFLICT(address) DO UPDATE SET balance = accounts.balance + excluded.balance",
                (to_address, amount),
            )
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            tx_id = f"eth-{now}"
            cur.execute(
                "INSERT INTO transactions(tx_id, from_addr, to_addr, amount, token, timestamp, memo) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (tx_id, sender, to_address, amount, "ETH", now, ""),
            )
            result = {"from": sender, "to": to_address, "amount": amount, "sender_balance": sender_balance - amount}
            self._audit(conn, "transfer_eth", result)
            self._store_idempotent(conn, idempotency_key, result)
            conn.commit()
            self._snapshot()
            conn.close()
            return result

    async def swap_tokens(
        self,
        token_in: str,
        token_out: str,
        amount: float,
        address: str | None = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        logger.info("swap_tokens actor=%s in=%s out=%s amount=%s", address, token_in, token_out, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            cur = conn.cursor()
            actor = address or self._default_actor(conn)
            token_in_u = token_in.upper()
            token_out_u = token_out.upper()
            cur.execute(
                "SELECT balance FROM token_balances WHERE address = ? AND token = ?", (actor, token_in_u)
            )
            row = cur.fetchone()
            in_balance = float(row["balance"]) if row else 0.0
            if in_balance < amount:
                conn.close()
                raise ValueError("Insufficient token balance.")
            price_in = await self.get_token_price(token_in_u)
            price_out = await self.get_token_price(token_out_u)
            amount_out = round(amount * price_in / max(price_out, 1e-8), 6)
            cur.execute(
                "UPDATE token_balances SET balance = balance - ? WHERE address = ? AND token = ?",
                (amount, actor, token_in_u),
            )
            cur.execute(
                "INSERT INTO token_balances(address, token, balance) VALUES(?, ?, ?) "
                "ON CONFLICT(address, token) DO UPDATE SET balance = token_balances.balance + excluded.balance",
                (actor, token_out_u, amount_out),
            )
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            tx_id = f"swap-{now}"
            cur.execute(
                "INSERT INTO transactions(tx_id, from_addr, to_addr, amount, token, timestamp, memo) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (tx_id, actor, actor, amount, token_in_u, now, f"swap to {token_out_u}"),
            )
            result = {"token_in": token_in_u, "token_out": token_out_u, "amount_out": amount_out, "actor": actor}
            self._audit(conn, "swap_tokens", result)
            self._store_idempotent(conn, idempotency_key, result)
            conn.commit()
            self._snapshot()
            conn.close()
            return result

    async def approve_token(self, owner: str, spender: str, amount: float, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        logger.info("approve_token owner=%s spender=%s amount=%s", owner, spender, amount)
        if amount < 0:
            raise ValueError("Amount must be non-negative.")
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO approvals(owner, spender, amount) VALUES(?, ?, ?) "
                "ON CONFLICT(owner, spender) DO UPDATE SET amount = excluded.amount",
                (owner, spender, amount),
            )
            result = {"owner": owner, "spender": spender, "amount": amount}
            self._audit(conn, "approve_token", result)
            self._store_idempotent(conn, idempotency_key, result)
            conn.commit()
            self._snapshot()
            conn.close()
            return result

    async def revoke_approval(self, owner: str, spender: str, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        logger.info("revoke_approval owner=%s spender=%s", owner, spender)
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO approvals(owner, spender, amount) VALUES(?, ?, 0) "
                "ON CONFLICT(owner, spender) DO UPDATE SET amount = 0",
                (owner, spender),
            )
            result = {"owner": owner, "spender": spender, "amount": 0.0}
            self._audit(conn, "revoke_approval", result)
            self._store_idempotent(conn, idempotency_key, result)
            conn.commit()
            self._snapshot()
            conn.close()
            return result

    async def simulate_transaction(self, to: str, value: float, data_field: str | None = None) -> Dict[str, Any]:
        """交易预执行，简单规则：检查余额与黑名单。"""
        logger.info("simulate_transaction to=%s value=%s", to, value)
        async with self._lock:
            conn = self._conn()
            sender = self._default_actor(conn)
            cur = conn.execute("SELECT balance FROM accounts WHERE address = ?", (sender,))
            row = cur.fetchone()
            sender_balance = float(row["balance"]) if row else 0.0
            rep = await self.check_address_reputation(to)
            conn.close()
            return {
                "sender": sender,
                "can_afford": sender_balance >= value,
                "target_reputation": rep,
                "estimated_fee": 0.001,
                "note": "blacklist" if rep and rep != "unknown" else "ok",
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

    async def bridge_asset(self, token: str, target_chain: str, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        logger.info("bridge_asset token=%s target=%s", token, target_chain)
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            record = {"token": token.upper(), "to_chain": target_chain, "timestamp": now}
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO bridges(id, token, to_chain, timestamp) VALUES(?, ?, ?, ?)",
                (f"bridge-{uuid4()}", record["token"], record["to_chain"], record["timestamp"]),
            )
            self._audit(conn, "bridge_asset", record)
            self._store_idempotent(conn, idempotency_key, record)
            conn.commit()
            self._snapshot()
            conn.close()
            return record

    async def stake_tokens(self, protocol: str, amount: float, idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        logger.info("stake_tokens protocol=%s amount=%s", protocol, amount)
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        async with self._lock:
            conn = self._conn()
            cached = self._idempotent_result(conn, idempotency_key)
            if cached is not None:
                conn.close()
                return cached
            actor = self._default_actor(conn)
            now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            record = {"actor": actor, "protocol": protocol, "amount": amount, "timestamp": now}
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO stakes(id, actor, protocol, amount, timestamp) VALUES(?, ?, ?, ?, ?)",
                (f"stake-{uuid4()}", actor, protocol, amount, now),
            )
            self._audit(conn, "stake_tokens", record)
            self._store_idempotent(conn, idempotency_key, record)
            conn.commit()
            self._snapshot()
            conn.close()
            return {"actor": actor, "protocol": protocol, "amount": amount}

    # ---------- Helpers ----------
    def _default_actor(self, conn: sqlite3.Connection) -> str:
        # Legacy default actor; stored in meta in JSON, so fallback to treasury
        return os.getenv("DEFAULT_ACTOR") or "treasury"
