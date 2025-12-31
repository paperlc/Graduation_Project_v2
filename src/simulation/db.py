"""SQLite helpers for the simulated ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS accounts (
    address TEXT PRIMARY KEY,
    balance REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS token_balances (
    address TEXT,
    token TEXT,
    balance REAL DEFAULT 0,
    PRIMARY KEY (address, token)
);

CREATE TABLE IF NOT EXISTS approvals (
    owner TEXT,
    spender TEXT,
    amount REAL DEFAULT 0,
    PRIMARY KEY (owner, spender)
);

CREATE TABLE IF NOT EXISTS contracts (
    address TEXT PRIMARY KEY,
    owner TEXT,
    bytecode TEXT
);

CREATE TABLE IF NOT EXISTS ens (
    domain TEXT PRIMARY KEY,
    address TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    token TEXT PRIMARY KEY,
    price REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reputations (
    address TEXT PRIMARY KEY,
    status TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    from_addr TEXT,
    to_addr TEXT,
    amount REAL,
    token TEXT,
    timestamp TEXT,
    memo TEXT
);

CREATE TABLE IF NOT EXISTS liquidity_pools (
    pool TEXT PRIMARY KEY,
    token0 TEXT,
    token1 TEXT,
    liquidity_usd REAL,
    fee_bps INTEGER
);

CREATE TABLE IF NOT EXISTS bridges (
    id TEXT PRIMARY KEY,
    token TEXT,
    to_chain TEXT,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS stakes (
    id TEXT PRIMARY KEY,
    actor TEXT,
    protocol TEXT,
    amount REAL,
    timestamp TEXT
);

CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    result TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT,
    payload TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def load_initial_state(
    conn: sqlite3.Connection,
    accounts: dict,
    token_balances: dict,
    approvals: dict,
    contracts: dict,
    ens: dict,
    prices: dict,
    reputations: dict,
    transactions: Iterable[dict],
    liquidity_pools: dict,
):
    cur = conn.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO accounts(address, balance) VALUES(?, ?)",
        [(addr, float(balance)) for addr, balance in accounts.items()],
    )
    rows = []
    for addr, tokens in token_balances.items():
        for token, bal in tokens.items():
            rows.append((addr, token.upper(), float(bal)))
    cur.executemany(
        "INSERT OR REPLACE INTO token_balances(address, token, balance) VALUES(?, ?, ?)",
        rows,
    )
    rows = []
    for owner, spends in approvals.items():
        for spender, amt in spends.items():
            rows.append((owner, spender, float(amt)))
    cur.executemany(
        "INSERT OR REPLACE INTO approvals(owner, spender, amount) VALUES(?, ?, ?)",
        rows,
    )
    cur.executemany(
        "INSERT OR REPLACE INTO contracts(address, owner, bytecode) VALUES(?, ?, ?)",
        [(addr, data.get("owner", ""), data.get("bytecode", "")) for addr, data in contracts.items()],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO ens(domain, address) VALUES(?, ?)",
        [(domain, addr) for domain, addr in ens.items()],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO prices(token, price) VALUES(?, ?)",
        [(token, float(price)) for token, price in prices.items()],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO reputations(address, status) VALUES(?, ?)",
        [(addr.lower(), status) for addr, status in reputations.items()],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO transactions(tx_id, from_addr, to_addr, amount, token, timestamp, memo) VALUES(?, ?, ?, ?, ?, ?, ?)",
        [
            (
                tx.get("tx"),
                tx.get("from"),
                tx.get("to"),
                float(tx.get("amount", 0.0)),
                tx.get("token", "ETH"),
                tx.get("timestamp", ""),
                tx.get("memo", ""),
            )
            for tx in transactions
        ],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO liquidity_pools(pool, token0, token1, liquidity_usd, fee_bps) VALUES(?, ?, ?, ?, ?)",
        [
            (pool, data.get("token0"), data.get("token1"), float(data.get("liquidity_usd", 0.0)), int(data.get("fee_bps", 0)))
            for pool, data in liquidity_pools.items()
        ],
    )
    conn.commit()
