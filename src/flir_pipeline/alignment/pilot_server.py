"""Same-machine review surface; distinct endpoint and ledger from Human Audit v1.1."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def make_server(store, port=8766, ui_name="pilot_ui.html"):
    if ui_name not in {"pilot_ui.html", "dependency_ui.html"}:
        raise ValueError("Unknown review surface")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def allowed(self):
            hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            host = self.headers.get("Host")
            return (
                host in hosts
                and self.headers.get("Origin", f"http://{host}") == f"http://{host}"
            )

        def respond(self, code, data, mime="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, b"{}")
            path = urlsplit(self.path).path
            try:
                if path == "/":
                    return self.respond(
                        200,
                        Path(__file__).with_name(ui_name).read_bytes(),
                        "text/html; charset=utf-8",
                    )
                if path == "/api/pilot":
                    return self.respond(
                        200, json.dumps(store.snapshot(), ensure_ascii=False).encode()
                    )
                if path == "/api/report":
                    return self.respond(
                        200, json.dumps(store.report(), ensure_ascii=False).encode()
                    )
                if path in {"/api/export/json", "/api/export/csv"}:
                    fmt = path.rsplit("/", 1)[1]
                    return self.respond(
                        200,
                        store.export(fmt),
                        "application/json"
                        if fmt == "json"
                        else "text/csv; charset=utf-8",
                    )
                if path.startswith("/media/"):
                    return self.respond(
                        200, store.media(unquote(path[len("/media/") :])), "image/png"
                    )
                return self.respond(404, b"{}")
            except (ValueError, KeyError, OSError) as exc:
                return self.respond(400, json.dumps({"error": str(exc)}).encode())

        def do_POST(self):
            if (
                not self.allowed()
                or self.headers.get("X-Alignment-Review") != "1"
                or self.headers.get("Content-Type") != "application/json"
            ):
                return self.respond(403, b"{}")
            if self.path != "/api/decisions":
                return self.respond(404, b"{}")
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Invalid request size")
                result = store.save(json.loads(self.rfile.read(size)))
                return self.respond(
                    201, json.dumps(result, ensure_ascii=False).encode()
                )
            except (ValueError, KeyError, OSError) as exc:
                return self.respond(
                    409 if "conflict" in str(exc).lower() else 400,
                    json.dumps({"error": str(exc)}).encode(),
                )

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
