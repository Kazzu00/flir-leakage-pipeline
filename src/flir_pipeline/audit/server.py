"""Loopback-only review API, bounded media, explicit browser origin allowlist."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .store import Store


def make_server(store: Store, port=8765, frontend_origin="http://127.0.0.1:5173"):
    origin = urlsplit(frontend_origin)
    if (
        origin.scheme != "http"
        or origin.hostname not in {"127.0.0.1", "localhost"}
        or origin.path
    ):
        raise ValueError("Frontend origin must be a loopback HTTP origin")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Avoid logging private event/query identifiers.

        def allowed(self):
            hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            return (
                self.headers.get("Host") in hosts
                and self.headers.get("Origin", frontend_origin) == frontend_origin
            )

        def respond(self, code, data, mime="application/json", download=None):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{download}"'
                )
            if self.headers.get("Origin") == frontend_origin:
                self.send_header("Access-Control-Allow-Origin", frontend_origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            if not self.allowed():
                return self.respond(403, b"{}")
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", frontend_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers", "Content-Type, X-Audit-Review"
            )
            self.end_headers()

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, b"{}")
            try:
                path = urlsplit(self.path).path
                if path == "/api/audit":
                    return self.respond(
                        200, json.dumps(store.snapshot(), ensure_ascii=False).encode()
                    )
                if path in {"/api/export/json", "/api/export/csv"}:
                    fmt = path.rsplit("/", 1)[1]
                    return self.respond(
                        200,
                        store.export(fmt),
                        "application/json"
                        if fmt == "json"
                        else "text/csv; charset=utf-8",
                        download=f"audit-decisions.{fmt}",
                    )
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[:2] == ["api", "media"]:
                    data, mime = store.media(parts[2], int(parts[3]))
                    return self.respond(200, data, mime)
                self.respond(404, b"{}")
            except (ValueError, KeyError, OSError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode())

        def do_POST(self):
            if (
                not self.allowed()
                or self.headers.get("X-Audit-Review") != "1"
                or self.headers.get("Content-Type") != "application/json"
            ):
                return self.respond(403, b"{}")
            if self.path != "/api/decisions":
                return self.respond(404, b"{}")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                result = store.save_decision(json.loads(self.rfile.read(length)))
                self.respond(201, json.dumps(result).encode())
            except (ValueError, KeyError) as exc:
                self.respond(
                    409 if "conflict" in str(exc).lower() else 400,
                    json.dumps({"error": str(exc)}).encode(),
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
