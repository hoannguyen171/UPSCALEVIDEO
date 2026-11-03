"""Định vị ffmpeg/ffprobe + chạy subprocess an toàn (không qua shell).

Thứ tự tìm: biến môi trường -> PATH -> imageio-ffmpeg (Colab/Kaggle) -> ./bin trong repo.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_EXE = ".exe" if os.name == "nt" else ""

FFMPEG = None
FFPROBE = None


def _existing(path):
    return bool(path) and Path(path).exists()


def _candidates(name, env_key):
    env = os.environ.get(env_key)
    if env:
        yield env
    found = shutil.which(name)
    if found:
        yield found
    if name == "ffmpeg":
        try:  # Colab/Kaggle: gói imageio-ffmpeg mang sẵn binary
            import imageio_ffmpeg

            yield imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    yield str(Path(__file__).resolve().parent.parent / "bin" / (name + _EXE))


def _resolve(name, env_key):
    for candidate in _candidates(name, env_key):
        if _existing(candidate):
            return candidate
    return None


def refresh_paths():
    """(Re)dò lại ffmpeg/ffprobe, ví dụ sau khi apt-get install trong notebook."""
    global FFMPEG, FFPROBE
    FFMPEG = _resolve("ffmpeg", "VALO_FFMPEG")
    FFPROBE = _resolve("ffprobe", "VALO_FFPROBE")
    return FFMPEG, FFPROBE


def set_paths(ffmpeg=None, ffprobe=None):
    """Ép đường dẫn thủ công (nếu người dùng biết chính xác binary)."""
    global FFMPEG, FFPROBE
    if ffmpeg:
        FFMPEG = str(ffmpeg)
    if ffprobe:
        FFPROBE = str(ffprobe)
    return FFMPEG, FFPROBE


def require_ffmpeg():
    if not FFMPEG:
        refresh_paths()
    if not FFMPEG:
        raise RuntimeError(
            "Không tìm thấy ffmpeg. Cài bằng: apt-get install -y ffmpeg (Colab/Kaggle), "
            "hoặc đặt biến môi trường VALO_FFMPEG=/duong/dan/ffmpeg."
        )
    return FFMPEG


def require_ffprobe():
    if not FFPROBE:
        refresh_paths()
    if not FFPROBE:
        raise RuntimeError(
            "Không tìm thấy ffprobe. Cài bằng: apt-get install -y ffmpeg, "
            "hoặc đặt biến môi trường VALO_FFPROBE=/duong/dan/ffprobe."
        )
    return FFPROBE


def run(cmd, timeout=None, capture=True, on_stdout_line=None):
    """Chạy lệnh, trả (returncode, stdout, stderr). Không bao giờ dùng shell=True.

    stderr được ghi ra file tạm thay vì PIPE: trên Windows buffer pipe chỉ 4 KB, nếu chỉ
    đọc stdout thì ffmpeg sẽ chặn cứng ở stderr (deadlock) khi log vượt buffer.
    """
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errfile:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=errfile,
            text=True,
            errors="replace",
            bufsize=1,
        )
        stdout_lines = []
        try:
            if not capture:
                proc.wait(timeout=timeout)
            elif on_stdout_line:
                for line in proc.stdout:
                    stdout_lines.append(line)
                    on_stdout_line(line)
                proc.wait(timeout=timeout)
            else:
                out, _ = proc.communicate(timeout=timeout)
                stdout_lines.append(out or "")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"Lệnh vượt quá thời gian chờ ({timeout}s): {Path(cmd[0]).name}")
        errfile.seek(0)
        stderr = errfile.read()
    return proc.returncode, "".join(stdout_lines), stderr or ""


refresh_paths()
