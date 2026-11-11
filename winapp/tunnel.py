"""Cầu nối Kaggle → app Windows: chia sẻ video qua HTTP + tunnel công khai (cloudflared quick tunnel).

Kaggle **không** cho kết nối vào notebook, nên chiều "app → Kaggle" phải đi qua một URL công khai:
app mở 1 HTTP server nội bộ, phơi nó ra bằng cloudflared (không cần tài khoản), rồi nhét URL đó vào
worker chạy trên Kaggle. Chiều "Kaggle → app" đi qua ``kaggle kernels output`` (tải file kết quả về).
"""

from __future__ import annotations

import os
import platform
import re
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_RELEASES = "https://github.com/cloudflare/cloudflared/releases/latest/download"
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


class TunnelError(RuntimeError):
    pass


def platform_asset():
    """Tên file cloudflared theo hệ điều hành hiện tại."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system.startswith("win"):
        return f"cloudflared-windows-{arch}.exe", "cloudflared.exe"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz", "cloudflared"
    return f"cloudflared-linux-{arch}", "cloudflared"


def parse_url(text):
    """Lấy URL trycloudflare trong log của cloudflared."""
    match = TUNNEL_URL_RE.search(text or "")
    return match.group(0) if match else None


def ensure_cloudflared(dest_dir, log=None, url=None):
    """Tải cloudflared (1 file ~40 MB) nếu chưa có. Trả đường dẫn."""
    asset, local_name = platform_asset()
    target = Path(dest_dir) / local_name
    if target.exists() and target.stat().st_size > 1 << 20:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    source = url or f"{CLOUDFLARED_RELEASES}/{asset}"
    if log:
        log(f"Đang tải cloudflared: {source}")
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(source, timeout=180) as response, open(tmp, "wb") as handle:
        total = int(response.headers.get("content-length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if log and total and done % (5 << 20) < (1 << 20):
                log(f"  cloudflared {done / 1048576:.0f}/{total / 1048576:.0f} MB")
    if asset.endswith(".tgz"):
        import tarfile  # noqa: PLC0415

        with tarfile.open(tmp) as archive:
            archive.extractall(target.parent)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(target)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if log:
        log(f"cloudflared sẵn sàng: {target}")
    return target


class QuickTunnel:
    """Một quick tunnel đang chạy (URL công khai → ``http://127.0.0.1:<port>``)."""

    def __init__(self, proc, url, log_path=None):
        self.proc = proc
        self.url = url
        self.log_path = log_path

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        """Dừng cloudflared.

        Trên Windows, nếu ``exe`` là một shim ``.cmd``/``.bat`` thì tiến trình thật là **con** của nó:
        ``terminate()`` chỉ giết shim và để tiến trình con sống tiếp. Vì vậy phải giết cả cây.
        """
        if self.alive:
            proc = self.proc
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, check=False)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        # đóng file log nếu tunnel này mở nó (nếu không, Windows giữ file và không xoá được thư mục)
        handle = getattr(self, "log_handle", None)
        if handle is not None and hasattr(handle, "close"):
            try:
                handle.close()
            except Exception:                      # noqa: BLE001 - dọn dẹp, không được ném lỗi
                pass
            self.log_handle = None


def start_quick_tunnel(port, exe, timeout=120, log=None, log_path=None):
    """Chạy ``cloudflared tunnel --url http://127.0.0.1:<port>`` và đợi URL công khai."""
    cmd = [str(exe), "tunnel", "--url", f"http://127.0.0.1:{int(port)}", "--no-autoupdate"]
    handle = open(log_path, "wb") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            bufsize=1)
    found = {}
    ready = threading.Event()

    def reader():
        try:
            for line in proc.stdout:
                if handle is not subprocess.DEVNULL:
                    handle.write(line.encode("utf-8", "replace"))
                    handle.flush()
                url = parse_url(line)
                if url and not found.get("url"):
                    found["url"] = url
                    ready.set()
        except Exception:
            pass
        finally:
            ready.set()

    threading.Thread(target=reader, name="cloudflared-log", daemon=True).start()
    if not ready.wait(timeout):
        proc.kill()
        raise TunnelError(f"cloudflared không trả URL công khai sau {timeout}s (xem {log_path})")
    if not found.get("url"):
        proc.kill()
        raise TunnelError("cloudflared thoát mà không có URL — kiểm tra mạng/firewall")
    if log:
        log(f"Tunnel công khai: {found['url']}")
    tunnel = QuickTunnel(proc, found["url"], log_path=str(log_path) if log_path else None)
    tunnel.log_handle = handle          # để stop() đóng file log (Windows giữ file nếu chưa đóng)
    return tunnel


def wait_public(url, timeout=60, interval=2.0):
    """Đợi URL công khai thực sự truy cập được (tunnel mất vài giây để lên)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=15) as response:
                if response.status == 200 and response.read().decode("utf-8", "replace").strip() == "ok":
                    return True
        except Exception:
            time.sleep(interval)
    return False
