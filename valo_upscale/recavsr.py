"""ReCaVSR (int8) trong ComfyUI — one-step streaming video super-resolution 4x trên Wan2.2.

Node pack: https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR (GPL-3.0) — dùng **node core của ComfyUI**
(``LoadVideo``/``SaveVideo``) + 2 node của pack: ``ReCaVSRLoader`` và ``ReCaVSRUpscale``.
Workflow dựng theo đúng file chính thức ``example_workflows/recavsr_video_4x_api.json``.

Weight (đặt vào ``ComfyUI/models/recavsr/``; node tự đăng ký thư mục này):

| File | Nguồn | Kích thước |
|---|---|---|
| ``transformer_int8_convrot.safetensors`` (**int8**, mặc định) | ``PedroMarinhoDev/ReCaVSR-ComfyUI`` (``recavsr/…``) | 6,27 GB |
| ``transformer.safetensors`` (bf16) | ``kopper/ReCaVSR`` | 10,6 GB |
| ``prompt.safetensors`` | ``kopper/ReCaVSR`` | 4,2 MB |
| ``flashdecoder.safetensors`` | ``kopper/ReCaVSR`` | 229 MB |

Lưu ý cho T4 (Kaggle): int8 ConvRot cần **compute capability ≥ 7.5** → T4 (sm_75) vừa đủ; ``kernels="fast"``
cần CC 8.0 cho decoder nhưng **có bản eager** nên vẫn chạy được (chậm hơn). VRAM tham chiếu của tác giả:
~11,6 GB ở 1440×1440 với int8 (vừa T4 16 GB), ~21 GB ở 3840×1920 (không vừa → node tự hạ cấu hình/kv_offload
hoặc báo lỗi nêu rõ ``scale`` lớn nhất chạy được).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import probe
from .comfy_upscale import ComfyError, _hf_token, check_ready as _check_ready, find_local_weight, link_or_copy

NODE_PACK_URL = "https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR"
NODE_PACK_DIRNAME = "ComfyUI-ReCaVSR"

# Node cần cho graph ReCaVSR (2 node của pack + 2 node video core của ComfyUI)
RECAVSR_NODES = ("ReCaVSRLoader", "ReCaVSRUpscale")
CORE_NODES = ("LoadVideo", "SaveVideo")

TRANSFORMERS = {
    "int8": {
        "repo": "PedroMarinhoDev/ReCaVSR-ComfyUI",
        "remote": "recavsr/transformer_int8_convrot.safetensors",
        "name": "transformer_int8_convrot.safetensors",
        "size_gb": 6.27,
        "note": "int8 ConvRot: 6,27 GB, transformer nhanh ~1,8×, cần GPU CC ≥ 7.5 (T4 vừa đủ)",
    },
    "bf16": {
        "repo": "kopper/ReCaVSR",
        "remote": "transformer.safetensors",
        "name": "transformer.safetensors",
        "size_gb": 10.6,
        "note": "bf16 gốc: 10,6 GB, chất lượng tham chiếu",
    },
}
SIDE_FILES = (
    ("kopper/ReCaVSR", "prompt.safetensors", "prompt.safetensors"),
    ("kopper/ReCaVSR", "flashdecoder.safetensors", "flashdecoder.safetensors"),
)
ENCODERS = ("auto", "hevc nvenc", "h264 nvenc", "h264")
COLOR_FIXES = ("none", "adain", "wavelet")
DECODERS = ("flashdecoder", "vae")
KERNELS = ("fast", "exact")


def resolve_precision(kind):
    key = (kind or "int8").strip().lower()
    if key in ("int8", "convrot", "int8_convrot"):
        return "int8"
    if key in ("bf16", "fp16", "bf16s"):
        return "bf16"
    raise ComfyError(f"precision phải là 'int8' hoặc 'bf16', không phải {kind!r}.")


def transformer_name(kind="int8"):
    return TRANSFORMERS[resolve_precision(kind)]["name"]


# ------------------------------------------------------------------ weight


def download_weights(models_dir, kind="int8", token=None, source_dir=None, log=None):
    """Tải transformer (int8/bf16) + prompt + flashdecoder vào ``models/recavsr``."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ComfyError("Thiếu huggingface_hub. Cài: pip install -q huggingface_hub") from exc

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token(token)
    plan = [(TRANSFORMERS[resolve_precision(kind)]["repo"],
             TRANSFORMERS[resolve_precision(kind)]["remote"],
             TRANSFORMERS[resolve_precision(kind)]["name"])] + list(SIDE_FILES)

    paths = []
    for repo_id, remote, name in plan:
        target = models_dir / name
        if target.exists() and target.stat().st_size > 0:
            if log:
                log(f"đã có sẵn: {target}")
            paths.append(str(target))
            continue
        local = find_local_weight(source_dir, name)
        if local is not None:
            how = link_or_copy(local, target)
            if log:
                log(f"{how} từ {local} → {target}")
            paths.append(str(target))
            continue
        if log:
            log(f"Đang tải {remote} từ {repo_id} {'(có HF_TOKEN)' if token else '(không xác thực)'} ...")
        paths.append(hf_hub_download(repo_id=repo_id, filename=remote, local_dir=str(models_dir), token=token))
    return paths


def weights_ready(models_dir, kind="int8"):
    """Đủ transformer + 2 file phụ chưa (kiểm cả dung lượng để tránh file rỗng)."""
    models_dir = Path(models_dir)
    names = [transformer_name(kind)] + [name for _, _, name in SIDE_FILES]
    return all((models_dir / name).exists() and (models_dir / name).stat().st_size > 1024 for name in names)


def install_node_commands(comfy_dir, url=NODE_PACK_URL, python=None, with_requirements=True):
    """Lệnh clone node pack vào ``custom_nodes`` và cài requirements của nó (triton, opencv…)."""
    import sys  # noqa: PLC0415

    python = python or sys.executable
    custom_nodes = Path(comfy_dir) / "custom_nodes"
    target = custom_nodes / NODE_PACK_DIRNAME
    commands = []
    if not (target / ".git").exists():
        commands.append(["git", "clone", "--depth", "1", "-q", url, str(target)])
    else:
        commands.append(["git", "-C", str(target), "pull", "--ff-only", "-q"])
    if with_requirements:
        requirements = target / "requirements.txt"
        if requirements.exists():
            commands.append([python, "-m", "pip", "install", "-q", "-r", str(requirements)])
    return commands


# ------------------------------------------------------------------ workflow


def build_workflow(video_file, transformer, *, scale=4.0, color_fix="none", seed=42, decoder="flashdecoder",
                   kernels="fast", kv_offload=False, encoder="auto", save_format="auto", save_codec="auto",
                   filename_prefix="recavsr/video"):
    """Graph API cho ReCaVSR — đúng cấu trúc ``recavsr_video_4x_api.json`` chính thức.

    ``video_file`` là tên file trong thư mục input của ComfyUI.
    """
    if color_fix not in COLOR_FIXES:
        raise ComfyError(f"color_fix phải thuộc {COLOR_FIXES}")
    if decoder not in DECODERS:
        raise ComfyError(f"decoder phải thuộc {DECODERS}")
    if kernels not in KERNELS:
        raise ComfyError(f"kernels phải thuộc {KERNELS}")
    if encoder not in ENCODERS:
        raise ComfyError(f"encoder phải thuộc {ENCODERS}")
    if not (0.5 <= float(scale) <= 8.0):
        raise ComfyError("scale phải trong [0.5, 8] (model huấn luyện ở 4)")

    upscale = {
        "model": ["3", 0],
        "video": ["1", 0],
        "scale": float(scale),
        "color_fix": color_fix,
        "seed": int(seed),
        "decoder": decoder,
        "kernels": kernels,
        "kv_offload": bool(kv_offload),
        "encoder": encoder,
    }
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_file}},
        "3": {"class_type": "ReCaVSRLoader", "inputs": {"transformer": transformer}},
        "4": {"class_type": "ReCaVSRUpscale", "inputs": upscale},
        "6": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": filename_prefix,
                                                     "format": save_format, "codec": save_codec}},
    }


def check_ready(base, transformer, log=None):
    """Kiểm node của pack + file transformer có trong danh sách của ``ReCaVSRLoader``."""
    state = _check_ready(base, required_nodes=CORE_NODES + RECAVSR_NODES,
                         extra_combos=(("ReCaVSRLoader", "transformer", transformer),))
    if log and not state["ok"]:
        log(f"ReCaVSR chưa sẵn sàng: node thiếu {state['missing_nodes']} | vấn đề {state['problems']}")
    return state


# ------------------------------------------------------------------ chạy


def upscale_video(input_path, output_path, *, comfy_dir, base="http://127.0.0.1:8188", scale=4.0,
                  precision="int8", transformer=None, color_fix="none", kernels="fast", decoder="flashdecoder",
                  kv_offload=False, encoder="auto", seed=42, timeout=7200, progress=None, stop_check=None,
                  **kwargs):
    """Upscale 1 clip bằng ReCaVSR; trả dict kết quả (giống ``comfy_upscale.upscale_video``)."""
    from .comfy_upscale import upscale_video as _comfy_upscale_video  # noqa: PLC0415

    name = transformer or transformer_name(precision)
    state = check_ready(base, name)
    if not state["ok"]:
        raise ComfyError("ReCaVSR chưa sẵn sàng: " + "; ".join(state["missing_nodes"] + list(state["problems"].values()))
                         + f". Cần ComfyUI ≥ 0.44, node pack {NODE_PACK_DIRNAME} trong custom_nodes, "
                           f"và {name} + prompt.safetensors + flashdecoder.safetensors trong models/recavsr.")

    if progress:
        progress(2, "preparing", f"ReCaVSR {name} • scale {scale} • kernels {kernels} • encoder {encoder}")

    def builder(video_file, model_file, vae_file):
        return build_workflow(video_file, name, scale=scale, color_fix=color_fix, seed=seed, decoder=decoder,
                              kernels=kernels, kv_offload=kv_offload, encoder=encoder)

    return _comfy_upscale_video(
        input_path, output_path, comfy_dir=comfy_dir, base=base, scale=float(scale),
        model_file=name, vae_file=None, workflow_builder=builder, timeout=timeout, progress=progress,
        stop_check=stop_check, required_nodes=CORE_NODES + RECAVSR_NODES, **kwargs,
    )


def describe(kind="int8"):
    """Thông tin ngắn để in ra UI/log."""
    entry = TRANSFORMERS[resolve_precision(kind)]
    return {"precision": resolve_precision(kind), **{k: entry[k] for k in ("repo", "remote", "name", "size_gb", "note")}}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="valo_upscale.recavsr",
                                     description="ReCaVSR (int8) 4x video SR qua ComfyUI")
    parser.add_argument("--input")
    parser.add_argument("--out")
    parser.add_argument("--comfy-dir", default=str(Path.cwd() / "ComfyUI"))
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--precision", default="int8", choices=["int8", "bf16"])
    parser.add_argument("--color-fix", default="none", choices=list(COLOR_FIXES))
    parser.add_argument("--kernels", default="fast", choices=list(KERNELS))
    parser.add_argument("--decoder", default="flashdecoder", choices=list(DECODERS))
    parser.add_argument("--encoder", default="auto", choices=list(ENCODERS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-offload", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--weights-source", default=None)
    parser.add_argument("--download", action="store_true", help="Tải weight rồi thoát")
    parser.add_argument("--check", action="store_true", help="Kiểm node/weight rồi thoát")
    parser.add_argument("--info", action="store_true", help="In thông tin model rồi thoát")
    args = parser.parse_args(argv)

    log = lambda text: print(text, flush=True)  # noqa: E731
    if args.info:
        print(json.dumps([describe(k) for k in ("int8", "bf16")], ensure_ascii=False, indent=2))
        return 0
    if args.download:
        for path in download_weights(Path(args.comfy_dir) / "models" / "recavsr", args.precision,
                                     token=args.token, source_dir=args.weights_source, log=log):
            print("đã có:", path)
        return 0
    if args.check:
        state = check_ready(args.server, transformer_name(args.precision), log=log)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state["ok"] else 1
    if not (args.input and args.out):
        print("Cần --input và --out (hoặc --check/--download/--info)", flush=True)
        return 2
    result = upscale_video(args.input, args.out, comfy_dir=args.comfy_dir, base=args.server, scale=args.scale,
                           precision=args.precision, color_fix=args.color_fix, kernels=args.kernels,
                           decoder=args.decoder, encoder=args.encoder, seed=args.seed, kv_offload=args.kv_offload,
                           progress=lambda percent, stage, text: log(f"  [{percent:3d}%] {text}"))
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
