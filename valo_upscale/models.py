"""Registry model Real-ESRGAN: tải weight chính thức + tự suy ra kiến trúc từ state_dict.

Bảng model trùng với app gốc (3 model trong thư mục models/ của realesrgan-ncnn-vulkan)
cộng thêm realesrgan-x2plus để có 2x "thật" cho ảnh.

Weight tải từ GitHub Releases của xinntao/Real-ESRGAN (BSD-3-Clause).
"""

import os
import re
import sys
from math import isqrt
from pathlib import Path

DEFAULT_WEIGHTS_DIR = Path(
    os.environ.get("VALO_WEIGHTS_DIR", Path.home() / ".cache" / "valo-upscale" / "weights")
)

_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"

MODEL_REGISTRY = {
    # key: dùng đúng tên model của app gốc
    "realesrgan-x4plus": {
        "urls": [f"{_RELEASE}/v0.1.0/RealESRGAN_x4plus.pth"],
        "file": "RealESRGAN_x4plus.pth",
        "native": 4,
        "label": "realesrgan-x4plus — ảnh chụp, phong cảnh, vật thể",
    },
    "realesrgan-x4plus-anime": {
        "urls": [f"{_RELEASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"],
        "file": "RealESRGAN_x4plus_anime_6B.pth",
        "native": 4,
        "label": "realesrgan-x4plus-anime — anime, manga, tranh vẽ 2D",
    },
    "realesr-animevideov3": {
        "urls": [f"{_RELEASE}/v0.2.5.0/realesr-animevideov3.pth"],
        "file": "realesr-animevideov3.pth",
        "native": 4,
        "label": "realesr-animevideov3 — tối ưu video, xử lý nhanh",
    },
    "realesrgan-x2plus": {
        "urls": [f"{_RELEASE}/v0.2.1/RealESRGAN_x2plus.pth"],
        "file": "RealESRGAN_x2plus.pth",
        "native": 2,
        "label": "realesrgan-x2plus — phóng 2x thật (bonus, không có trong app gốc)",
    },
}

# Model mặc định theo loại media, giống app gốc
DEFAULT_IMAGE_MODEL = "realesrgan-x4plus"
DEFAULT_VIDEO_MODEL = "realesr-animevideov3"

ALIASES = {
    "realesrgan_x4plus": "realesrgan-x4plus",
    "realesrgan-x4plus-anime-6b": "realesrgan-x4plus-anime",
    "realesrgan_x4plus_anime_6b": "realesrgan-x4plus-anime",
    "realesr-animevideov3-x4": "realesr-animevideov3",
    "animevideov3": "realesr-animevideov3",
    "realesrganx4plus": "realesrgan-x4plus",
    "x4plus": "realesrgan-x4plus",
    "x2plus": "realesrgan-x2plus",
}


def resolve_model_name(name):
    """Chuẩn hoá tên model; trả về key trong MODEL_REGISTRY (raise nếu không có)."""
    if not name:
        return None
    key = name.strip().lower().replace(".pth", "")
    key = ALIASES.get(key, key)
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Model không hỗ trợ: {name}. Chọn một trong: {', '.join(MODEL_REGISTRY)}"
        )
    return key


def native_scale(model_name):
    return MODEL_REGISTRY[resolve_model_name(model_name)]["native"]


def weights_dir():
    DEFAULT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_WEIGHTS_DIR


def weight_path(model_name):
    entry = MODEL_REGISTRY[resolve_model_name(model_name)]
    return weights_dir() / entry["file"]


def download_weights(model_name, progress=None):
    """Tải weight nếu chưa có. progress(percent|None, text)."""
    key = resolve_model_name(model_name)
    entry = MODEL_REGISTRY[key]
    target = weights_dir() / entry["file"]
    if target.exists() and target.stat().st_size > 0:
        return target

    import requests  # noqa: PLC0415  (chỉ cần khi thực sự tải)

    last_error = None
    for url in entry["urls"]:
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "valo-upscale/1.0"}) as res:
                res.raise_for_status()
                total = int(res.headers.get("content-length") or 0)
                done = 0
                with open(tmp, "wb") as fh:
                    for chunk in res.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if progress and total:
                            progress(
                                int(done * 100 / total),
                                f"Đang tải {entry['file']}: {done / 1048576:.1f}/{total / 1048576:.1f} MB",
                            )
            tmp.replace(target)
            if progress:
                progress(100, f"Đã tải {entry['file']}")
            return target
        except Exception as exc:  # thử URL dự phòng
            last_error = exc
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    raise RuntimeError(f"Không tải được weight {entry['file']}: {last_error}")


def infer_arch(state):
    """Suy kiến trúc trực tiếp từ state_dict -> dict cho archs.RRDBNet / archs.SRVGGNetCompact."""
    keys = [k for k in state.keys() if k.endswith("weight")]

    if any(".rdb" in k for k in keys):
        blocks = [int(k.split(".")[1]) for k in keys if k.startswith("body.") and k.split(".")[1].isdigit()]
        first = state["conv_first.weight"]
        in_ch, num_feat = first.shape[1], first.shape[0]
        scale = {3: 4, 12: 2, 48: 1}.get(in_ch)
        if scale is None:
            raise ValueError(f"RRDBNet có số kênh vào lạ: {in_ch}")
        return {
            "kind": "rrdb",
            "scale": scale,
            "num_feat": num_feat,
            "num_block": (max(blocks) + 1) if blocks else 23,
            "num_grow_ch": 32,
        }

    convs = sorted(
        int(m.group(1))
        for m in (re.fullmatch(r"body\.(\d+)\.weight", k) for k in keys)
        if m and state[f"body.{m.group(1)}.weight"].dim() == 4
    )
    if not convs:
        raise ValueError("Không nhận dạng được kiến trúc từ state_dict (không có key RRDB/SRVGG).")

    num_feat = state["body.0.weight"].shape[0]
    out_ch = state[f"body.{convs[-1]}.weight"].shape[0]
    upscale = isqrt(out_ch // 3)
    if upscale * upscale * 3 != out_ch:
        raise ValueError(f"SRVGGNetCompact có số kênh ra lạ: {out_ch}")
    return {
        "kind": "srvgg",
        "num_feat": num_feat,
        "num_conv": len(convs) - 2,
        "upscale": upscale,
        "act_type": "prelu" if f"body.1.weight" in state else "relu",
    }


def build_network(arch):
    from . import archs

    if arch["kind"] == "rrdb":
        net = archs.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=arch["scale"],
            num_feat=arch["num_feat"],
            num_block=arch["num_block"],
            num_grow_ch=arch["num_grow_ch"],
        )
        divisor, out_scale = arch["scale"], arch["scale"]
    else:
        net = archs.SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=arch["num_feat"],
            num_conv=arch["num_conv"],
            upscale=arch["upscale"],
            act_type=arch["act_type"],
        )
        divisor, out_scale = 1, arch["upscale"]

    # divisor: RRDBNet scale<4 dùng pixel_unshuffle nên cạnh ảnh phải chia hết cho hệ số này
    net.divisor = divisor
    net.out_scale = out_scale
    return net


def load_model(model_name, device="cpu", half=False, progress=None):
    """Tải weight (nếu cần), dựng mạng, nạp state_dict strict. Trả (net, native_scale)."""
    import torch  # noqa: PLC0415

    key = resolve_model_name(model_name)
    path = download_weights(key, progress=progress)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for wrapper in ("params_ema", "params", "state_dict"):
            if wrapper in state and isinstance(state[wrapper], dict):
                state = state[wrapper]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"File weight không hợp lệ: {path}")

    arch = infer_arch(state)
    net = build_network(arch)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Weight {path.name} không khớp kiến trúc {arch['kind']} "
            f"(thiếu {len(missing)} key, thừa {len(unexpected)} key: {unexpected[:3]})."
        )
    native = arch.get("scale") or arch["upscale"]
    net.eval().to(device)
    if half:
        net.half()
    return net, native


def list_models():
    for key, entry in MODEL_REGISTRY.items():
        mark = "✔ đã có" if (weights_dir() / entry["file"]).exists() else "… chưa tải"
        print(f"  {key:<24} native {entry['native']}x  [{mark}]  {entry['label']}")


if __name__ == "__main__":  # pragma: no cover - tiện dụng khi chạy trực tiếp
    print(f"Thư mục weight: {weights_dir()}")
    list_models()
    if len(sys.argv) > 1:
        load_model(sys.argv[1], progress=lambda p, t: print(f"  [{p}%] {t}"))
        print(f"Nạp thành công: {sys.argv[1]}")
