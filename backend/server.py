"""Local inference API; requires an explicit secret, never trains from live traffic."""
import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core import Model, decode_flatbuffer


class InferenceServer(HTTPServer):
    # Deliberately single-threaded: bounded memory/concurrency, suitable for local baseline inference.
    def __init__(self, address, key, model=None):
        super().__init__(address, Handler)
        self.key = key
        self.model = model


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format, *args):
        pass  # Never log request headers, keys or player data.

    def respond(self, status, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != '/health':
            return self.respond(404, {'error': 'not_found'})
        self.respond(200, {'status': 'ok', 'ready': self.server.model is not None})

    def do_POST(self):
        if self.path != '/predict_binary':
            return self.respond(404, {'error': 'not_found'})
        supplied = self.headers.get('X-Secret-Key', '')
        if not hmac.compare_digest(supplied.encode(), self.server.key.encode()):
            return self.respond(401, {'error': 'unauthorized'})
        if self.headers.get('Transfer-Encoding'):
            return self.respond(400, {'error': 'content_length_required'})
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/octet-stream':
            return self.respond(415, {'error': 'expected_binary'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 8 <= length <= 65536:
                return self.respond(413, {'error': 'invalid_size'})
            data = self.rfile.read(length)
            frames = decode_flatbuffer(data)
        except (ValueError, TypeError, TimeoutError, OSError):
            return self.respond(400, {'error': 'invalid_movement_window'})
        model = self.server.model
        if model is None:
            return self.respond(503, {'ready': False, 'error': 'model_not_trained'})
        score = model.score(frames)
        self.respond(200, {'ready': True, 'confidence': score * 100,
                           'threshold': model.threshold * 100,
                           'suspicious': score > model.threshold})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    key = os.environ.get('ANGUARD_API_KEY', '')
    if len(key) < 24:
        parser.exit(1, 'Set ANGUARD_API_KEY to a random secret of at least 24 characters.\n')
    # Plain HTTP is restricted to loopback; remote deployment requires a separate TLS transport change.
    model = Model(args.model) if args.model else None
    server = InferenceServer(('127.0.0.1', args.port), key, model)
    print(f'Inference listening on loopback port {args.port}; model ready={model is not None}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
