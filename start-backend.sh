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
If system Python is unsupported (e.g. 3.14), downloads a private Python 3.12 via uv.
System Python is left unchanged. No Java is needed for the API.
Default: 127.0.0.1:8000. Ctrl+C stops the API. No trained model is fabricated.
The API key is retained privately in backend/.local/api-key unless SILENTAC_API_KEY (or legacy ANGUARD_API_KEY) is set.
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
managed_python() {
    local bootstrap_python=''
    local candidate
    for candidate in python3 python3.14 python3.13 python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then
            bootstrap_python="$(command -v "$candidate")"
            break
        fi
    done
    if [[ -z "$bootstrap_python" ]]; then
        install_python
        bootstrap_python="$(command -v python3)"
    fi
    local bootstrap_dir="$BACKEND_DIR/.local/bootstrap"
    if [[ ! -x "$bootstrap_dir/bin/python" ]]; then
        if ! "$bootstrap_python" -m venv "$bootstrap_dir"; then
            install_python
            "$bootstrap_python" -m venv "$bootstrap_dir"
        fi
    fi
    local uv_bin="$bootstrap_dir/bin/uv"
    if [[ ! -x "$uv_bin" ]] || [[ "$("$uv_bin" --version)" != 'uv 0.9.6' ]]; then
        "$bootstrap_dir/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: 'uv==0.9.6'
    fi
    export UV_PYTHON_INSTALL_DIR="$BACKEND_DIR/.local/python"
    export UV_PYTHON_BIN_DIR="$BACKEND_DIR/.local/bin"
    export PATH="$UV_PYTHON_BIN_DIR:$PATH"
    export UV_CACHE_DIR="$BACKEND_DIR/.local/uv-cache"
    export UV_NATIVE_TLS=true
    echo 'Installing private Python 3.12 (system Python will not be changed)...'
    "$uv_bin" python install 3.12
    PYTHON_BIN="$("$uv_bin" python find --managed-python 3.12)"
}

if [[ "${ANGUARD_MANAGED_PYTHON:-0}" == 1 ]] || ! find_python; then
    managed_python
fi

if [[ -x "$VENV_DIR/bin/python" ]] && ! "$VENV_DIR/bin/python" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] < (3,14)))'; then
    backup="$VENV_DIR.backup-$(date +%s)-$$"
    echo "Preserving incompatible environment in $backup"
    mv -- "$VENV_DIR" "$backup"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo 'Creating local Python environment...'
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
        managed_python
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
