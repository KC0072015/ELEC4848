#!/usr/bin/env bash
# start.sh — Launch API + Frontend (Linux / macOS)
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
  echo "ERROR: virtualenv not found at $VENV"
  echo "       Run 'poetry install' first."
  exit 1
fi

echo "Starting API on http://localhost:8000 ..."
cd "$ROOT/backend"
"$VENV/bin/uvicorn" api:app --reload --port 8000 &
API_PID=$!

echo "Starting Frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "Both services running. Press Ctrl+C to stop."
trap "echo ''; echo 'Shutting down...'; kill $API_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
wait
