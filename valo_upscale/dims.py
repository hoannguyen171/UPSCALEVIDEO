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


# ---------------------------------------------------------------- VRAM ReCaVSR
# Hai điểm tham chiếu của tác giả ReCaVSR (int8 + FlashDecoder, fast kernels):
#   ~11,6 GB ở 1440x1440  (2,07 MP)   |   ~21 GB ở 3840x1920  (7,37 MP)
# Khớp luỹ thừa qua 2 điểm: GB = 8,2483 * MP^0,46774  (tăng chậm hơn tuyến tính)
VRAM_A = 8.2483
VRAM_B = 0.46774
VRAM_BF16_EXTRA_GB = 4.0          # bản bf16 nặng hơn int8 khoảng 4 GB
T4_VRAM_GB = 16.0                 # Kaggle T4 (16 GB, dùng được ~15)


def estimate_recavsr_vram_gb(width, height, scale=4.0, precision="int8"):
    """Ước lượng VRAM cho 1 clip ReCaVSR. Trả ``(gb, out_w, out_h)``.

    Chỉ để cảnh báo trước — con số thật còn phụ thuộc nội dung/khung hình.
    """
    out_w, out_h = calculate_video_dimensions(width, height, scale)
    megapixels = out_w * out_h / 1e6
    gb = VRAM_A * (megapixels ** VRAM_B)
    if str(precision).lower().startswith("bf"):
        gb += VRAM_BF16_EXTRA_GB
    return round(gb, 1), out_w, out_h


def vram_warning(width, height, scale=4.0, precision="int8", budget_gb=T4_VRAM_GB):
    """Cảnh báo tiếng Việt khi ước lượng vượt VRAM; ``None`` nếu vừa."""
    gb, out_w, out_h = estimate_recavsr_vram_gb(width, height, scale, precision)
    if gb <= budget_gb:
        return None, gb, (out_w, out_h)
    text = (f"Ước lượng ~{gb:g} GB > {budget_gb:g} GB VRAM của T4 ở {out_w}x{out_h} ({precision}, scale {scale:g}) "
            f"→ dễ hết VRAM.")
    # tìm mức scale nhỏ nhất còn vừa (dimension bị clamp 4K nên scale 4 và 2 có thể ra cùng kích thước)
    for smaller in (2.0, 1.0, 0.5):
        if smaller >= float(scale):
            continue
        fit_gb, fit_w, fit_h = estimate_recavsr_vram_gb(width, height, smaller, precision)
        if fit_gb <= budget_gb:
            text += f" Nên dùng --scale {smaller:g} (~{fit_gb:g} GB, {fit_w}x{fit_h})."
            break
    else:
        text += " Kể cả scale nhỏ nhất vẫn vượt — hãy giảm độ phân giải video, hoặc bật --kv-offload."
    if precision == "bf16":
        text += " Cũng có thể dùng --precision int8 (nhẹ hơn ~4 GB)."
    return text, gb, (out_w, out_h)
