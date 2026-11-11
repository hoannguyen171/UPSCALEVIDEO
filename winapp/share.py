"""HTTP server nội bộ chỉ phục vụ **một file video** cho worker trên Kaggle tải về.

Có ``/health`` để app (và worker) kiểm tra tunnel đã sống chưa, và ``?t=<token>`` để URL không
dùng được nếu bị lộ ngoài ý muốn.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

CHUNK = 1 << 20


class _Handler(BaseHTTPRequestHandler):
    server_version = "ValoShare/1.0"

    def log_message(self, *args):  # im lặng
        pass

    def _authorized(self):
        token = self.server.share_token
        if not token:
            return True
        provided = parse_qs(urlparse(self.path).query).get("t", [""])[0]
        return provided == token

    def _info(self):
        return (f"valo-share: {self.server.share_file.name} "
                f"({self.server.share_file.stat().st_size} bytes)").encode()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/", "/info"):
            body = self._info()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._authorized():
            self.send_error(401, "missing or wrong token")
            return
        if path not in ("/video", f"/{self.server.share_file.name}"):
            self.send_error(404, "only /video is served")
            return
        size = self.server.share_file.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", f'attachment; filename="{self.server.share_file.name}"')
        self.end_headers()
        with open(self.server.share_file, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.server.bytes_sent += len(chunk)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            return
        if not self._authorized() or path not in ("/video", f"/{self.server.share_file.name}"):
            if not self._authorized():
                self.send_error(401, "missing or wrong token")
            else:
                self.send_error(404, "only /video is served")
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(self.server.share_file.stat().st_size))
        self.end_headers()


class FileShareServer:
    """Chia sẻ 1 file qua HTTP; ``url`` là địa chỉ nội bộ, ghép với URL tunnel để ra địa chỉ công khai."""

    def __init__(self, path, host="127.0.0.1", port=0, token=None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"không thấy file để chia sẻ: {self.path}")
        self.host = host
        self.port = int(port)
        self.token = token
        self.bytes_sent = 0
        self._httpd = None

    def start(self):
        httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        httpd.share_file = self.path
        httpd.share_token = self.token
        httpd.bytes_sent = 0
        self._httpd = httpd
        self.port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return self.local_url

    @property
    def local_url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def video_url(self):
        query = f"?t={self.token}" if self.token else ""
        return f"{self.local_url}/video{query}"

    @property
    def bytes_served(self):
        return getattr(self._httpd, "bytes_sent", 0)

    def public_video_url(self, tunnel_url):
        query = f"?t={self.token}" if self.token else ""
        return f"{tunnel_url.rstrip('/')}/video{query}"

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
