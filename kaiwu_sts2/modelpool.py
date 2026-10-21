"""modelpool equivalent: minimal HTTP model file pool (kaiwu's Go service is
replaced by ~60 lines of stdlib; interface kept: /latest, /upload, /download).

Storage layout: {MODEL_DIR}/model_{version}.zip + latest.txt
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import get

_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "Sts2ModelPool/0.1"

    @property
    def store(self) -> Path:
        return Path(get("MODEL_DIR"))

    def _version_file(self) -> Path:
        return self.store / "latest.txt"

    def _current_version(self) -> int:
        f = self._version_file()
        return int(f.read_text().strip()) if f.exists() else 0

    def _send_json(self, obj: dict, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter logs, kaiwu-style
        print(f"[modelpool] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/latest":
            self._send_json({"version": self._current_version()})
        elif parsed.path == "/download":
            version = int(parse_qs(parsed.query)["version"][0])
            f = self.store / f"model_{version}.zip"
            if not f.exists():
                self._send_json({"error": f"model v{version} not found"}, 404)
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self._send_json({"error": "not found"}, 404)
            return
        version = int(parse_qs(parsed.query)["version"][0])
        length = int(self.headers["Content-Length"])
        data = self.rfile.read(length)
        self.store.mkdir(parents=True, exist_ok=True)
        (self.store / f"model_{version}.zip").write_bytes(data)
        with _LOCK:
            if version > self._current_version():
                self._version_file().write_text(str(version))
        self._send_json({"status": "ok", "version": version})

    def log_message(self, fmt, *args):  # quiet default access log
        pass


def serve() -> None:
    port = int(get("MODELPOOL_PORT"))
    Path(get("MODEL_DIR")).mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[modelpool] serving on 0.0.0.0:{port}, dir={get('MODEL_DIR')}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    serve()
