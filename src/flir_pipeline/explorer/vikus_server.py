"""Loopback-only, read-only static serving of a verified generated bundle."""

from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from flir_pipeline.explorer.vikus import verify_bundle

CSP = ("default-src 'self'; script-src 'self' 'unsafe-eval'; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "connect-src 'self'; font-src 'self'; worker-src 'self' blob:; "
       "object-src 'none'; base-uri 'self'; frame-ancestors 'none'")


def local_server(directory: Path, workspace: Path, port: int = 8765) -> ThreadingHTTPServer:
    base = (workspace.resolve() / "reports/explorer/vikus").resolve()
    directory = directory.resolve()
    if not directory.is_relative_to(base) or directory == base:
        raise ValueError("Serve only a generated bundle under reports/explorer/vikus")
    receipt = verify_bundle(directory)
    allowed = {*receipt["output_sha256"], "bundle_receipt.json"}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def send_head(self):
            host = self.headers.get("Host", "").partition(":")[0]
            if host not in {"127.0.0.1", "localhost"}:
                self.send_error(403, "Local host required")
                return None
            requested = unquote(urlsplit(self.path).path).lstrip("/") or "index.html"
            path = (directory / requested).resolve()
            if requested not in allowed or not path.is_relative_to(directory):
                self.send_error(404)
                return None
            return super().send_head()

        def end_headers(self):
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, *_):
            # Image/content IDs need not fill terminal or persistent access logs.
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve_bundle(directory: Path, workspace: Path, port: int = 8765) -> None:
    with local_server(directory, workspace, port) as server:
        print(f"VIKUS local: http://127.0.0.1:{server.server_port} · Ctrl+C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
