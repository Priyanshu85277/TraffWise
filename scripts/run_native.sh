#!/usr/bin/env bash
# Run TraffWise natively (without Docker).
#
# Usage:
#   ./scripts/run_native.sh backend          # set up venv, install deps, start FastAPI on :8000
#   ./scripts/run_native.sh backend --gpu    # same, but install CUDA (cu128) PyTorch wheels
#   ./scripts/run_native.sh frontend         # npm install + start React dev server on :3000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  grep '^#   ' "$0" | sed 's/^#   //'
  exit 1
}

run_backend() {
  local requirements="requirements-cpu.txt"
  if [[ "${1:-}" == "--gpu" ]]; then
    requirements="requirements.txt"
  fi

  cd "$ROOT/backend"

  if [[ ! -d .venv ]]; then
    echo "Creating virtualenv at backend/.venv"
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  pip install -r "$requirements"

  echo "Checking runtime assets..."
  python "$ROOT/scripts/download_assets.py"

  echo "Starting backend at http://localhost:8000 (docs at /docs)"
  python server.py
}

run_frontend() {
  cd "$ROOT/frontend"
  if [[ ! -d node_modules ]]; then
    npm ci
  fi
  echo "Starting frontend at http://localhost:3000"
  npm start
}

case "${1:-}" in
  backend)  shift; run_backend "$@" ;;
  frontend) run_frontend ;;
  *)        usage ;;
esac
