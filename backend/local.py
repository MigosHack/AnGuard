"""Start the loopback API with a retained local secret, without putting it in arguments/logs."""
import os
from pathlib import Path
import secrets
import sys

state = Path(__file__).resolve().parent / '.local'
state.mkdir(mode=0o700, exist_ok=True)
key_file = state / 'api-key'
if not os.environ.get('ANGUARD_API_KEY'):
    try:
        with os.fdopen(os.open(key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as stream:
            stream.write(secrets.token_urlsafe(32))
    except FileExistsError:
        if key_file.is_symlink():
            raise SystemExit('Refusing symlink for local API key')
    os.environ['ANGUARD_API_KEY'] = key_file.read_text().strip()
os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('server.py')), *sys.argv[1:]])
