"""Tính kích thước đích khi upscale — port 1:1 từ app gốc (src/upscaler.js).

- Video: clamp tối đa 4K UHD (3840x2160 ngang / 2160x3840 dọc) để H.264 giải mã được.
- Ảnh: clamp tối đa 8192 px mỗi cạnh.
Cả hai đều ép cạnh chia hết cho 2 (yuv420p) và giữ nguyên tỉ lệ khung hình.
"""

MIN_VIDEO_DIM = 128
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192


def calculate_video_dimensions(orig_w, orig_h, scale):
    w = int(orig_w) if int(orig_w or 0) else 640
    h = int(orig_h) if int(orig_h or 0) else 360

    target_w = round(w * scale)
    target_h = round(h * scale)

    max_w = 3840 if w >= h else 2160
    max_h = 2160 if w >= h else 3840

    if target_w > max_w or target_h > max_h:
        factor = min(max_w / target_w, max_h / target_h)
        target_w = round(target_w * factor)
        target_h = round(target_h * factor)

    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2

    target_w = max(MIN_VIDEO_DIM, target_w)
    target_h = max(MIN_VIDEO_DIM, target_h)
    return target_w, target_h


def calculate_image_dimensions(orig_w, orig_h, scale):
    w = int(orig_w) if int(orig_w or 0) else 1024
    h = int(orig_h) if int(orig_h or 0) else 768

    target_w = round(w * scale)
    target_h = round(h * scale)

    if target_w > MAX_IMAGE_DIM or target_h > MAX_IMAGE_DIM:
        factor = min(MAX_IMAGE_DIM / target_w, MAX_IMAGE_DIM / target_h)
        target_w = round(target_w * factor)
        target_h = round(target_h * factor)

    target_w = (target_w // 2) * 2
    target_h = (target_h // 2) * 2

    target_w = max(MIN_IMAGE_DIM, target_w)
    target_h = max(MIN_IMAGE_DIM, target_h)
    return target_w, target_h
