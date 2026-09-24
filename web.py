"""Localhost web arayüzü: görsel yükle -> numara var mı, olasılık, okunan numara.

    python web.py            # http://localhost:8000
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from analyzer import Analyzer

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = open(os.path.join(HERE, "web", "index.html"), "rb").read()
MAX_BYTES = 25 * 1024 * 1024
LOCK = threading.Lock()   # tek model örneği, istekler sırayla
ANALYZER = None


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_BYTES:
            return self._send(413, json.dumps({"error": "Görsel boş ya da 25 MB'tan büyük."}).encode(), "application/json")
        data = self.rfile.read(n)
        t = time.time()
        try:
            with LOCK:
                res = ANALYZER.analyze(data)
            res["ms"] = int((time.time() - t) * 1000)
            self._send(200, json.dumps(res, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        except Exception as e:  # bozuk / desteklenmeyen dosya
            self._send(400, json.dumps({"error": f"Görsel işlenemedi: {e}"}, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        print("[web]", fmt % args)


def main():
    global ANALYZER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threshold", type=float, default=0.9)
    a = ap.parse_args()
    print("modeller yükleniyor...")
    ANALYZER = Analyzer(a.threshold)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"hazır: http://localhost:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
