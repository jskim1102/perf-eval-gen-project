from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


STATIC_ROOT = Path(__file__).resolve().parent / "static"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the dependency-free web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("FRONTEND_PORT"),
        required="FRONTEND_PORT" not in os.environ,
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=os.environ.get("BACKEND_PORT"),
        required="BACKEND_PORT" not in os.environ,
    )
    return parser.parse_args()


def handler_factory(backend_port: int):
    config_body = (
        "window.PERF_EVAL_CONFIG = "
        + json.dumps(
            {"backendPort": backend_port},
            separators=(",", ":"),
        )
        + ";\n"
    ).encode()

    class FrontendHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.split("?", 1)[0] == "/config.js":
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(config_body)))
                self.end_headers()
                self.wfile.write(config_body)
                return
            super().do_GET()

    FrontendHandler.config_body = config_body
    return FrontendHandler


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        handler_factory(args.backend_port),
    )
    print(
        f"frontend=http://{args.host}:{args.port} backend_port={args.backend_port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
