#!/usr/bin/env bash
set -euo pipefail

# Quick start helper: launch MCP (SSE) + Streamlit UI in one command.
# Usage: bash scripts/quickstart.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  echo "[info] Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MCP_TRANSPORT="${MCP_TRANSPORT:-sse}"

# Safe lane
MCP_HOST_SAFE="${MCP_HOST_SAFE:-127.0.0.1}"
MCP_PORT_SAFE="${MCP_PORT_SAFE:-8001}"
MCP_SSE_PATH_SAFE="${MCP_SSE_PATH_SAFE:-/sse}"
HEALTH_PORT_SAFE="${HEALTH_PORT_SAFE:-8081}"
LEDGER_DB_SAFE="${LEDGER_DB_SAFE:-$ROOT/data/ledger/ledger_safe.db}"
MCP_SERVER_URL_SAFE="${MCP_SERVER_URL_SAFE:-http://${MCP_HOST_SAFE}:${MCP_PORT_SAFE}${MCP_SSE_PATH_SAFE}}"

# Unsafe lane
MCP_HOST_UNSAFE="${MCP_HOST_UNSAFE:-127.0.0.1}"
MCP_PORT_UNSAFE="${MCP_PORT_UNSAFE:-8002}"
MCP_SSE_PATH_UNSAFE="${MCP_SSE_PATH_UNSAFE:-/sse}"
HEALTH_PORT_UNSAFE="${HEALTH_PORT_UNSAFE:-8082}"
LEDGER_DB_UNSAFE="${LEDGER_DB_UNSAFE:-$ROOT/data/ledger/ledger_unsafe.db}"
MCP_SERVER_URL_UNSAFE="${MCP_SERVER_URL_UNSAFE:-http://${MCP_HOST_UNSAFE}:${MCP_PORT_UNSAFE}${MCP_SSE_PATH_UNSAFE}}"

echo "[info] MCP transport=${MCP_TRANSPORT}"
echo "[info] SAFE   -> ${MCP_SERVER_URL_SAFE} ledger=${LEDGER_DB_SAFE}"
echo "[info] UNSAFE -> ${MCP_SERVER_URL_UNSAFE} ledger=${LEDGER_DB_UNSAFE}"

echo "[info] Resetting ledger DBs to seed from JSON baseline..."
mkdir -p "$(dirname "$LEDGER_DB_SAFE")" "$(dirname "$LEDGER_DB_UNSAFE")"
rm -f "$LEDGER_DB_SAFE" "$LEDGER_DB_UNSAFE"

echo "[mcp] Starting MCP server (safe)..."
LEDGER_DB="$LEDGER_DB_SAFE" MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_HOST_SAFE" MCP_PORT="$MCP_PORT_SAFE" MCP_SSE_PATH="$MCP_SSE_PATH_SAFE" HEALTH_PORT="$HEALTH_PORT_SAFE" \
  python -m src.simulation.server &
MCP_PID_SAFE=$!

echo "[mcp] Starting MCP server (unsafe)..."
LEDGER_DB="$LEDGER_DB_UNSAFE" MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_HOST_UNSAFE" MCP_PORT="$MCP_PORT_UNSAFE" MCP_SSE_PATH="$MCP_SSE_PATH_UNSAFE" HEALTH_PORT="$HEALTH_PORT_UNSAFE" \
  python -m src.simulation.server &
MCP_PID_UNSAFE=$!

trap 'echo "[mcp] Stopping MCP servers"; kill '"$MCP_PID_SAFE"' '"$MCP_PID_UNSAFE"' 2>/dev/null || true' EXIT

sleep 1
echo "[ui] Starting Streamlit at http://localhost:8501 ..."
MCP_SERVER_URL="$MCP_SERVER_URL_SAFE" MCP_SERVER_URL_SAFE="$MCP_SERVER_URL_SAFE" MCP_SERVER_URL_UNSAFE="$MCP_SERVER_URL_UNSAFE" \
  streamlit run app.py
