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
    _probe_cache.clear()
    return FFMPEG, FFPROBE


def set_paths(ffmpeg=None, ffprobe=None):
    """Ép đường dẫn thủ công (nếu người dùng biết chính xác binary)."""
    global FFMPEG, FFPROBE
    if ffmpeg:
        FFMPEG = str(ffmpeg)
    if ffprobe:
        FFPROBE = str(ffprobe)
    _probe_cache.clear()
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


# --------------------------------------------------------------- dò khả năng phần cứng

_probe_cache: dict[str, bool] = {}


def _probe(key, args):
    """Chạy thử 1 lệnh ffmpeg ngắn rồi nhớ kết quả (tránh dò lại mỗi job)."""
    if key in _probe_cache:
        return _probe_cache[key]
    ok = False
    try:
        code, _, _ = run([require_ffmpeg(), "-hide_banner", "-v", "error"] + args, timeout=60)
        ok = code == 0
    except Exception:
        ok = False
    _probe_cache[key] = ok
    return ok


def supports_encoder(name="h264_nvenc"):
    """GPU có NVENC dùng được không (T4/P100 trên Kaggle: có)."""
    return _probe(
        f"enc:{name}",
        ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=2", "-frames:v", "2", "-c:v", name, "-f", "null", "-"],
    )


def supports_cuda_decode():
    """NVDEC (giải mã bằng GPU) dùng được không.

    Lưu ý: ``-hwaccel cuda`` một mình KHÔNG đủ để kết luận — khi không có GPU, ffmpeg chỉ cảnh báo
    rồi tự giải mã bằng CPU và vẫn trả mã 0. Nên probe phải ép khung hình nằm trên GPU
    (``-hwaccel_output_format cuda`` + ``hwdownload``) và kiểm tra có dữ liệu ghi ra hay không.
    """
    if "dec:cuda" in _probe_cache:
        return _probe_cache["dec:cuda"]
    ok = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "probe.h264"
            dump = Path(tmp) / "probe.raw"
            code, _, _ = run(
                [require_ffmpeg(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=64x64:r=4",
                 "-frames:v", "4", "-c:v", "libx264", "-f", "h264", "-y", str(sample)],
                timeout=60,
            )
            if code == 0:
                code, _, _ = run(
                    [require_ffmpeg(), "-hide_banner", "-v", "error", "-hwaccel", "cuda",
                     "-hwaccel_output_format", "cuda", "-f", "h264", "-i", str(sample),
                     "-vf", "hwdownload,format=nv12", "-frames:v", "2", "-f", "rawvideo", "-y", str(dump)],
                    timeout=60,
                )
                ok = code == 0 and dump.exists() and dump.stat().st_size > 0
    except Exception:
        ok = False
    _probe_cache["dec:cuda"] = ok
    return ok


def pick_video_encoder(prefer="auto"):
    """'h264_nvenc' nếu dùng được, ngược lại 'libx264'."""
    if prefer in ("auto", "nvenc") and supports_encoder("h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


refresh_paths()
