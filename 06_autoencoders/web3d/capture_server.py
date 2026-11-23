"""Static server for the viewer, plus an endpoint it can post renders back to.

The page can save a render with a download link, but a browser refuses a run of
them: after the first one or two Chrome treats the rest as an unwanted burst and
drops them silently. The files also land in Downloads, not in the project. So
the page posts each render here instead and this writes it where it belongs.

    POST /capture?name=<stem>   body: the PNG bytes  ->  data/topography/renders/

Run through .claude/launch.json ("relief-viewer"), not by hand.
"""

import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 06_autoencoders
RENDERS = ROOT / "data" / "topography" / "renders"
SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self):
        if not self.path.startswith("/capture"):
            self.send_error(404)
            return
        _, _, query = self.path.partition("?")
        name = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("name", "render")
        name = "".join(c for c in name if c in SAFE) or "render"

        size = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(size)
        RENDERS.mkdir(parents=True, exist_ok=True)
        path = RENDERS / f"{name}.png"
        path.write_bytes(data)
        print(f"captured {path.relative_to(ROOT)}  ({len(data) / 1e6:.2f} MB)", flush=True)

        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):      # keep the console to the captures
        if "POST" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
    print(f"serving {ROOT} on {port}, captures -> {RENDERS}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
