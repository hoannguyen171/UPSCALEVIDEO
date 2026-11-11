"""Đo độ lệch A/V giữa video gốc và video đã upscale.

Dùng clip có **transient** (`valo-upscale-gpu/samples/sync_click.mp4`: ô trắng nhấp nháy đúng 1,000s và
bíp 1 kHz cùng lúc). Không dùng tương quan trên sine — sine là tuần hoàn nên đỉnh tương quan vô nghĩa.

    python check_av_sync.py samples/sync_click.mp4 <video_da_upscale>.mp4
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def audio_onset(path):
    """Thời điểm bíp (giây) — đỉnh bao hình của kênh audio."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "a.raw"
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-f", "s16le", "-ac", "1", "-ar", "16000", str(raw)])
        x = np.fromfile(raw, dtype=np.int16).astype(np.float64)
    if x.size == 0:
        raise SystemExit(f"{path}: không có audio")
    envelope = np.convolve(np.abs(x), np.ones(160) / 160, mode="same")
    return int(np.argmax(envelope)) / 16000, x.size / 16000


def video_flash_onset(path, threshold=0.015):
    """Thời điểm khung hình đổi cảnh mạnh đầu tiên sau khung 0 (giây) — ô trắng nhấp nháy."""
    out = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf",
                          f"select='gt(scene\\,{threshold})',showinfo", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    times = [float(line.split("pts_time:")[1].split()[0]) for line in out.splitlines() if "pts_time" in line]
    return times[0] if times else None      # khung đổi cảnh ĐẦU TIÊN = lúc ô trắng hiện ra


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    src, out = (Path(p) for p in argv)
    for tag, path in (("GỐC", src), ("KẾT QUẢ", out)):
        beep, dur = audio_onset(path)
        flash = video_flash_onset(path)
        print(f"{tag:8s} {path.name}: bíp {beep:.3f}s | khung trắng {flash if flash is None else f'{flash:.3f}s'} | dài {dur:.3f}s")
    src_beep, _ = audio_onset(src)
    src_flash = video_flash_onset(src)
    out_beep, _ = audio_onset(out)
    out_flash = video_flash_onset(out)
    if None in (src_flash, out_flash):
        raise SystemExit("không thấy khung trắng — clip phải là samples/sync_click.mp4")
    src_skew, out_skew = src_beep - src_flash, out_beep - out_flash
    print(f"lệch A/V trong bản gốc   : {src_skew:+.3f}s")
    print(f"lệch A/V trong bản kết quả: {out_skew:+.3f}s")
    print(f"==> upscale làm lệch thêm: {out_skew - src_skew:+.3f}s (0.000 = giữ nguyên đồng bộ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
