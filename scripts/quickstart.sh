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
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-8001}"
MCP_SSE_PATH="${MCP_SSE_PATH:-/sse}"
MCP_SERVER_URL="${MCP_SERVER_URL:-http://${MCP_HOST}:${MCP_PORT}${MCP_SSE_PATH}}"

echo "[info] MCP transport=${MCP_TRANSPORT} host=${MCP_HOST} port=${MCP_PORT} sse_path=${MCP_SSE_PATH}"
echo "[info] MCP_SERVER_URL=${MCP_SERVER_URL}"

echo "[mcp] Starting MCP server..."
MCP_TRANSPORT="$MCP_TRANSPORT" MCP_HOST="$MCP_HOST" MCP_PORT="$MCP_PORT" MCP_SSE_PATH="$MCP_SSE_PATH" \
  python -m src.simulation.server &
MCP_PID=$!
trap 'echo "[mcp] Stopping MCP server"; kill '"$MCP_PID"' 2>/dev/null || true' EXIT

sleep 1
echo "[ui] Starting Streamlit at http://localhost:8501 ..."
streamlit run app.py
