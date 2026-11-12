#!/usr/bin/env python3
"""Browser demo for the diabetes ANN viva.

Usage:
    cd diabetes_ann
    .venv/bin/python demo/app.py
Then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from diabetes_prediction.config import FEATURE_COLUMNS  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765
INDEX = Path(__file__).resolve().parent / "index.html"
MODEL = ROOT / "final_report" / "results" / "proposed_mlp_pipeline.joblib"

_bundle = joblib.load(MODEL)
PIPE = _bundle["pipeline"]
FEATURES = list(_bundle.get("feature_columns", FEATURE_COLUMNS))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self._send(404, b"Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            row = {name: float(payload[name]) for name in FEATURES}
            frame = pd.DataFrame([row], columns=FEATURES)
            proba = float(PIPE.predict_proba(frame)[0, 1])
            pred = int(proba >= 0.5)
            body = json.dumps({"probability": proba, "predicted_class": pred}).encode()
            self._send(200, body, "application/json")
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"error": str(exc)}).encode()
            self._send(400, body, "application/json")


def main() -> int:
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Demo ready: http://{HOST}:{PORT}", flush=True)
    print("Educational only — not a diagnostic tool. Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
