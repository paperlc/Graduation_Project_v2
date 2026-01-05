#!/usr/bin/env bash
set -euo pipefail

# Package the whole project into a zip archive.
# Usage: bash scripts/archive_project.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${ZIP_PATH:-$ROOT/Graduation_Project_v2.zip}"

if ! command -v zip >/dev/null 2>&1; then
  echo "[error] zip command is required but not found in PATH" >&2
  exit 1
fi

echo "[info] Packaging project at ${ROOT} -> ${ZIP_PATH}"
rm -f "$ZIP_PATH"

cd "$ROOT"
ZIP_BASENAME="$(basename "$ZIP_PATH")"

zip -r "$ZIP_PATH" . -x "$ZIP_BASENAME" "./$ZIP_BASENAME"

echo "[ok] Archive created at $ZIP_PATH"
