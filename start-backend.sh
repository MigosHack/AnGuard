#!/usr/bin/env bash
# Run from any directory. All retained files live beside this checkout.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$REPO_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash start-backend.sh [--port 8000] [--model path/to/model.json]
       bash start-backend.sh --install-only
       bash start-backend.sh --test
       bash start-backend.sh --train recordings/*.jsonl --output models/aim-v1.json

Creates backend/.venv, installs pinned dependencies and starts the local API.
On Debian/Ubuntu, installs Python/venv using apt if missing (root or sudo required).
Otherwise Python 3.10–3.13 with venv is required. No Java is needed for the API.
Default: 127.0.0.1:8000. Ctrl+C stops the API. No trained model is fabricated.
The API key is retained privately in backend/.local/api-key unless ANGUARD_API_KEY is set.
HELP
    exit 0
fi

install_python() {
    if ! command -v apt-get >/dev/null 2>&1; then
        echo 'Install Python 3.10–3.13 with venv, then rerun this script.' >&2
        exit 1
    fi
    local -a elevate=()
    if [[ "$EUID" -ne 0 ]]; then
        if ! command -v sudo >/dev/null 2>&1; then
            echo 'Python/venv is missing. Install python3 and python3-venv as administrator, then rerun.' >&2
            exit 1
        fi
        elevate=(sudo)
    fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv
}

PYTHON_BIN=''
find_python() {
    local candidate
    for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))' 2>/dev/null; then
            PYTHON_BIN="$(command -v "$candidate")"
            return 0
        fi
    done
    return 1
}
if ! find_python; then
    install_python
    find_python || { echo 'A supported Python 3.10–3.13 is still unavailable.' >&2; exit 1; }
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo 'Creating local Python environment...'
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        install_python
        find_python
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi
PYTHON="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/requirements.sha256"
EXPECTED="$("$PYTHON" - "$BACKEND_DIR/requirements.txt" <<'PY'
import hashlib, pathlib, sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes() + sys.version.encode()).hexdigest())
PY
)"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$EXPECTED" ]]; then
    echo 'Installing backend dependencies...'
    "$PYTHON" -m pip install --disable-pip-version-check --only-binary=:all: -r "$BACKEND_DIR/requirements.txt"
    "$PYTHON" -m pip check
    printf '%s\n' "$EXPECTED" > "$STAMP"
fi

case "${1:-}" in
    --install-only) echo 'Backend dependencies installed.'; exit 0 ;;
    --test)
        export PYTHONPATH="$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}"
        exec "$PYTHON" -m unittest discover -s "$BACKEND_DIR/tests" -v ;;
    --train) shift; exec "$PYTHON" "$BACKEND_DIR/train.py" "$@" ;;
esac
exec "$PYTHON" "$BACKEND_DIR/local.py" "$@"
