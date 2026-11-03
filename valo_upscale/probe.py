"""Đọc thông số media qua ffprobe — port từ app gốc (getMediaInfo trong src/upscaler.js).

Có xử lý metadata rotation (video quay dọc 90/270 độ): đổi chỗ width/height
để kích thước phản ánh đúng khung hình hiển thị.
"""

import json
import os
from pathlib import Path

from . import fftools

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".flv", ".wmv")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def media_kind(path):
    """'video' | 'image' theo phần mở rộng."""
    ext = Path(path).suffix.lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    return "video" if ext in VIDEO_EXT else "image"


def get_media_info(file_path):
    """Trả dict: width, height, fps, duration, nb_frames, has_audio, rotation, size, format."""
    path = str(Path(file_path).resolve())
    info = {
        "path": path,
        "width": 0,
        "height": 0,
        "fps": 24.0,
        "duration": 0.0,
        "nb_frames": 0,
        "has_audio": False,
        "rotation": 0,
        "size": os.path.getsize(path) if os.path.exists(path) else 0,
        "format": Path(path).suffix.lstrip(".").lower(),
    }
    try:
        ffprobe = fftools.require_ffprobe()
        code, out, _ = fftools.run(
            [
                ffprobe,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            timeout=60,
        )
        if code != 0 or not out.strip():
            return info
        data = json.loads(out)
    except Exception:
        return info

    streams = data.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {}) or {}

    info["has_audio"] = astream is not None
    info["format"] = fmt.get("format_name") or info["format"]

    if not vstream:
        return info

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    try:
        info["duration"] = float(fmt.get("duration") or vstream.get("duration") or 0)
    except (TypeError, ValueError):
        info["duration"] = 0.0
    try:
        info["nb_frames"] = int(vstream.get("nb_frames") or 0)
    except (TypeError, ValueError):
        info["nb_frames"] = 0

    rotation = 0
    tags = vstream.get("tags") or {}
    if tags.get("rotate") is not None:
        try:
            rotation = int(float(tags["rotate"]))
        except (TypeError, ValueError):
            rotation = 0
    else:
        for side in vstream.get("side_data_list") or []:
            if "rotation" in side:
                try:
                    rotation = int(float(side["rotation"]))
                except (TypeError, ValueError):
                    rotation = 0
                break
    if not rotation:
        ftags = fmt.get("tags") or {}
        if ftags.get("rotate") is not None:
            try:
                rotation = int(float(ftags["rotate"]))
            except (TypeError, ValueError):
                rotation = 0
    rotation = abs(rotation) % 360
    if rotation in (90, 270):
        width, height = height, width

    fps = 24.0
    rate = vstream.get("r_frame_rate") or ""
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            num_f, den_f = float(num), float(den)
            if den_f > 0 and num_f > 0:
                fps = round(num_f / den_f, 2)
        except ValueError:
            pass
    elif rate:
        try:
            fps = float(rate)
        except ValueError:
            pass

    info.update(width=width, height=height, fps=fps, rotation=rotation)
    if not info["nb_frames"] and info["duration"] > 0 and fps > 0:
        info["nb_frames"] = round(info["duration"] * fps)
    return info
