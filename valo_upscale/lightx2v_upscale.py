"""Upscale video bằng RelayVSR qua **LightX2V** — đường dùng được CẢ 2 GPU cho MỘT clip.

Khác đường ComfyUI (``comfy_upscale.py``): ComfyUI chạy 1 GPU cho 1 clip, muốn dùng 2 GPU thì phải mở
2 instance và chia clip (tăng thông lượng). LightX2V có config ``relayvsr_x2_sp.json`` với

    "parallel": {"seq_p_size": 2, "seq_p_attn_type": "ulysses"}

= **Ulysses sequence parallel**: chuỗi token của DiT chia đôi cho 2 GPU, attention trao đổi head↔sequence
bằng ``all2all`` → **một clip** chạy nhanh hơn và mỗi GPU giữ ít activation hơn.

Khởi động bằng ``torchrun`` (hoặc ``python -m torch.distributed.run``) với số process = ``seq_p_size × cfg_p_size``
— đúng như các script chính thức trong ``scripts/dist_infer/*.sh`` của LightX2V.

Lưu ý thực tế đã kiểm tra trong repo LightX2V (commit 9fd3d98):

* Weight: ``wan{suffix}.safetensors`` + ``flash{suffix}.safetensors`` (``_x2`` cho bản 2x,
  rỗng cho bản 4x) — xem ``lightx2v/models/networks/relayvsr/weights.py``.
* ``scripts/base/base.sh`` mặc định ``DTYPE=BF16`` → **trên T4 (Turing) phải đổi ``DTYPE=FP16``**.
* Runner RelayVSR đọc/ghi video bằng PyAV và **không** thấy xử lý audio → cần ghép lại tiếng từ video gốc
  (hàm :func:`ensure_audio`).
* ``requirements.txt`` của LightX2V rất nặng (``sgl-kernel``, ``qtorch``, ``torchao``, ``redis``, ``pymongo``…);
  đường cài gọn nhất là ``pip install -e .`` (theo ``pyproject.toml``) rồi bổ sung ``triton`` nếu thiếu.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import fftools, probe

REPO_URL = "https://github.com/ModelTC/LightX2V.git"

# scale -> (model_cls, repo HF, danh sách file cần tải, thư mục đích trong model_path)
WEIGHT_SOURCES = {
    2: ("relayvsr_x2", "lightx2v/RelayVSR_X2", ["wan_x2.safetensors", "flash_x2.safetensors"]),
    4: ("relayvsr_x4", "kopper/RelayVSR", ["wan.safetensors", "flash.safetensors"]),
}

CONFIG_MODES = ("single", "sp", "block_offload")


class LightX2VError(RuntimeError):
    """Lỗi khi chuẩn bị/chạy LightX2V (thông báo đã rõ cho người dùng)."""


def model_cls(scale):
    scale = int(scale)
    if scale not in WEIGHT_SOURCES:
        raise LightX2VError(f"RelayVSR qua LightX2V chỉ có 2 hoặc 4, không có {scale}x.")
    return WEIGHT_SOURCES[scale][0]


def config_path(repo_dir, scale=2, mode="sp"):
    """Đường dẫn config chính thức trong repo LightX2V."""
    scale = int(scale)
    if scale not in WEIGHT_SOURCES:
        raise LightX2VError(f"Hệ số không hợp lệ: {scale}")
    if mode not in CONFIG_MODES:
        raise LightX2VError(f"mode phải là một trong {CONFIG_MODES}, không phải {mode!r}.")
    suffix = "_x2" if scale == 2 else "_x4"
    extra = "" if mode == "single" else f"_{mode}"
    return Path(repo_dir) / "configs" / "relayvsr" / f"relayvsr{suffix}{extra}.json"


# ------------------------------------------------------------------ cài đặt / weight


def clone_repo(dest, url=REPO_URL, log=None):
    """Clone (hoặc pull) LightX2V. Trả đường dẫn repo."""
    dest = Path(dest)
    if (dest / ".git").exists():
        code, _, err = fftools.run(["git", "-C", str(dest), "pull", "--ff-only", "-q"])
        if log:
            log("đã cập nhật LightX2V" if code == 0 else f"pull lỗi (bỏ qua): {err[-200:]}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        code, _, err = fftools.run(["git", "clone", "--depth", "1", "-q", url, str(dest)])
        if code != 0:
            raise LightX2VError(f"clone LightX2V thất bại: {err[-300:]}")
        if log:
            log(f"đã clone LightX2V vào {dest}")
    return dest


def install_commands(repo_dir, python=None, full_requirements=False):
    """Danh sách lệnh pip để cài LightX2V.

    Mặc định cài theo ``pyproject.toml`` (``-e .``) và **bỏ qua** torch family để pip không thay
    bản CUDA sẵn có của Kaggle/Colab; ``full_requirements=True`` sẽ cài thêm ``requirements.txt``
    (nặng: ``sgl-kernel``, ``qtorch``, ``redis``, ``pymongo``…) — chỉ dùng khi thiếu module.
    """
    python = python or sys.executable
    repo_dir = Path(repo_dir)
    commands = [
        [python, "-m", "pip", "install", "-q", "--no-build-isolation", "-e", str(repo_dir)],
        [python, "-m", "pip", "install", "-q", "triton", "huggingface_hub"],
    ]
    if full_requirements:
        req = repo_dir / "requirements.txt"
        skip = ("torch", "torchvision", "torchaudio")
        filtered = repo_dir / "requirements-notorch.txt"
        lines = [line for line in req.read_text(encoding="utf-8").splitlines()
                 if line.split("==")[0].strip() not in skip]
        filtered.write_text("\n".join(lines), encoding="utf-8")
        commands.append([python, "-m", "pip", "install", "-q", "-r", str(filtered)])
    return commands


def download_weights(scale, model_dir, token=None, source_dir=None, repo_url=None, log=None):
    """Tải 2 checkpoint (Wan + Flash) vào ``model_dir`` đúng tên LightX2V mong đợi."""
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise LightX2VError("Thiếu huggingface_hub. Cài: pip install -q huggingface_hub") from exc

    from .comfy_upscale import _hf_token, find_local_weight, link_or_copy  # noqa: PLC0415

    scale = int(scale)
    _, repo_id, files = WEIGHT_SOURCES[scale]
    repo_id = repo_url or repo_id
    token = _hf_token(token)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for name in files:
        target = model_dir / name
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
            log(f"Đang tải {name} từ {repo_id} {'(có HF_TOKEN)' if token else '(không xác thực)'} ...")
        paths.append(hf_hub_download(repo_id=repo_id, filename=name, local_dir=str(model_dir), token=token))
    return paths


def weights_ready(scale, model_dir):
    """Đủ 2 file weight chưa (kiểm cả dung lượng > 1 MB để tránh file rỗng)."""
    _, _, files = WEIGHT_SOURCES[int(scale)]
    return all((Path(model_dir) / name).exists() and (Path(model_dir) / name).stat().st_size > 1 << 20
               for name in files)


# ------------------------------------------------------------------ config & lệnh


def prepare_config(repo_dir, scale=2, mode="sp", overrides=None, out_path=None):
    """Sao config chính thức rồi ghi đè vài khoá (giữ nguyên khối ``parallel`` khi mode='sp').

    Ví dụ hữu ích: ``{"video_codec": "libx264"}`` để player/browser đọc được ngay,
    hoặc ``{"use_compile": False}`` khi torch.compile lỗi trên môi trường lạ.
    """
    base = config_path(repo_dir, scale, mode)
    if not base.exists():
        raise LightX2VError(f"Không thấy config {base} — repo LightX2V đã clone chưa?")
    data = json.loads(base.read_text(encoding="utf-8"))
    for key, value in (overrides or {}).items():
        data[key] = value
    if mode == "sp" and "parallel" not in data:
        raise LightX2VError(f"Config {base.name} không có khối 'parallel' — chọn mode='sp' sai?")
    if out_path is None:
        return base
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def build_command(repo_dir, model_dir, video_path, save_path, *, scale=2, config=None, nproc=2, seed=42,
                  sr_ratio=None, python=None):
    """Lệnh chạy inference (đã đối chiếu với ``scripts/dist_infer/*.sh`` của LightX2V).

    ``nproc > 1`` → chạy bằng ``python -m torch.distributed.run --standalone`` (tương đương ``torchrun``,
    nhưng chắc chắn dùng đúng interpreter hiện tại).
    """
    python = python or sys.executable
    nproc = int(nproc)
    if nproc > 1:
        cmd = [python, "-m", "torch.distributed.run", f"--nproc_per_node={nproc}", "--standalone"]
    else:
        cmd = [python]
    cmd += [
        "-m", "lightx2v.infer",
        "--model_cls", model_cls(scale),
        "--task", "sr",
        "--model_path", str(model_dir),
        "--config_json", str(config or config_path(repo_dir, scale, "sp" if nproc > 1 else "single")),
        "--video_path", str(video_path),
        "--sr_ratio", str(sr_ratio if sr_ratio is not None else scale),
        "--save_result_path", str(save_path),
        "--seed", str(int(seed)),
    ]
    return cmd


def run_environment(repo_dir, gpus=None, dtype="FP16"):
    """Biến môi trường nên có khi chạy LightX2V (theo ``scripts/base/base.sh``)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["DTYPE"] = dtype  # T4 (Turing) không có BF16 → FP16
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in (gpus if isinstance(gpus, (list, tuple)) else [gpus]))
    return env


def check_environment(repo_dir, model_dir=None, scale=2, weights_model_dir=None):
    """Kiểm tra nhanh môi trường — trả dict để in ra cho người dùng biết còn thiếu gì."""
    report = {}
    report["python"] = sys.version.split()[0]
    try:
        import torch  # noqa: PLC0415

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        report["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(report["gpu_count"])]
    except Exception as exc:  # pragma: no cover
        report["torch"] = f"lỗi: {exc}"

    for module in ("triton", "av", "loguru", "diffusers", "transformers"):
        try:
            __import__(module)
            report[module] = True
        except Exception:
            report[module] = False

    report["ffmpeg"] = fftools.FFMPEG
    report["repo_exists"] = (Path(repo_dir) / "lightx2v").is_dir()
    report["configs"] = {mode: config_path(repo_dir, scale, mode).exists() for mode in CONFIG_MODES}
    model_dir = Path(model_dir or weights_model_dir or (Path(repo_dir) / "models" / f"RelayVSR_X{scale}"))
    report["model_dir"] = str(model_dir)
    _, _, files = WEIGHT_SOURCES[int(scale)]
    report["weights"] = {name: (model_dir / name).exists() for name in files}
    try:
        free = shutil.disk_usage(Path(repo_dir).anchor or ".").free
        report["disk_free_gb"] = round(free / 1073741824, 1)
    except Exception:
        report["disk_free_gb"] = None
    report["ok"] = bool(report.get("cuda_available")) and report["repo_exists"] and all(report["weights"].values())
    return report


# ------------------------------------------------------------------ chạy & hậu kiểm


def expected_size(width, height, ratio):
    """LightX2V làm tròn xuống bội số 2 (YUV420) theo README của họ."""
    return (int(width * ratio) // 2 * 2, int(height * ratio) // 2 * 2)


def ensure_audio(video_path, source_path, out_path=None, log=None):
    """Ghép lại audio từ video gốc nếu video kết quả không có tiếng.

    Runner RelayVSR của LightX2V đọc/ghi bằng PyAV và không thấy xử lý audio, nên bước này là bắt buộc
    để kết quả giống đường ComfyUI. Trả đường dẫn file có audio (hoặc None nếu không cần/không có nguồn).
    """
    info = probe.get_media_info(video_path)
    if info["has_audio"]:
        return None
    source = probe.get_media_info(source_path)
    if not source["has_audio"]:
        if log:
            log("video gốc không có audio — bỏ qua bước ghép tiếng")
        return None

    out_path = Path(out_path or Path(video_path).with_name(Path(video_path).stem + "_audio.mp4"))
    ffmpeg = fftools.require_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", str(video_path), "-i", str(source_path),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-shortest", str(out_path)]
    code, _, err = fftools.run(cmd, timeout=1800)
    if code != 0 or not out_path.exists():
        raise LightX2VError(f"Ghép audio thất bại (mã {code}): {err[-300:]}")
    if log:
        log(f"đã ghép audio từ video gốc → {out_path.name}")
    return out_path


def verify_output(output_path, video_path, ratio):
    """Kiểm kích thước/fps/số frame/audio của kết quả so với kỳ vọng."""
    got = probe.get_media_info(output_path)
    src = probe.get_media_info(video_path)
    want = expected_size(src["width"], src["height"], ratio)
    return {
        "info": got,
        "source": src,
        "expected_size": want,
        "size_ok": (got["width"], got["height"]) == want,
        "frames_ok": (not got["nb_frames"] or not src["nb_frames"]
                      or abs(got["nb_frames"] - src["nb_frames"]) <= 2),
        "audio_ok": got["has_audio"] or not src["has_audio"],
    }


def run_upscale(video_path, out_path, *, repo_dir, model_dir=None, scale=2, mode="sp", nproc=None, gpus=None,
                seed=42, sr_ratio=None, dtype="FP16", overrides=None, config=None, timeout=7200,
                keep_raw=False, progress=None, log_path=None, dry_run=False):
    """Chạy 1 clip RelayVSR bằng LightX2V rồi hậu kiểm (kích thước + audio)."""
    scale = int(scale)
    repo_dir = Path(repo_dir)
    model_dir = Path(model_dir or repo_dir / "models" / f"RelayVSR_X{scale}")
    log = progress or (lambda text: None)

    if nproc is None:
        nproc = 2 if mode == "sp" else 1
    if gpus is None and nproc > 1:
        gpus = list(range(nproc))

    if not config:
        if overrides:
            config = prepare_config(repo_dir, scale, mode, overrides,
                                    out_path=repo_dir / "configs" / "relayvsr" / f"valo_{mode}_x{scale}.json")
        else:
            config = prepare_config(repo_dir, scale, mode)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path if keep_raw else out_path.with_name(out_path.stem + "_noaudio.mp4")

    cmd = build_command(repo_dir, model_dir, Path(video_path), raw_path, scale=scale, config=config,
                        nproc=nproc, seed=seed, sr_ratio=sr_ratio)
    env = run_environment(repo_dir, gpus=gpus, dtype=dtype)

    if dry_run:
        return {"command": cmd, "env_subset": {k: env[k] for k in ("DTYPE", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")
                                               if k in env}, "config": str(config), "dry_run": True}

    log("Lệnh chạy: " + " ".join(str(part) for part in cmd))
    started = time.time()
    log_path = Path(log_path or out_path.with_suffix(".log"))
    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(cmd, cwd=str(repo_dir), env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise LightX2VError(f"LightX2V chạy quá {timeout}s — xem log {log_path}") from None

    tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-12:]
    if code != 0:
        raise LightX2VError(f"LightX2V thoát với mã {code}. 12 dòng cuối log:\n" + "\n".join(tail))
    if not raw_path.exists():
        raise LightX2VError(f"LightX2V chạy xong nhưng không thấy {raw_path}. Log:\n" + "\n".join(tail))

    if not keep_raw:
        log("LightX2V không giữ audio → ghép tiếng từ video gốc")
        remuxed = ensure_audio(raw_path, video_path, out_path=out_path, log=log)
        if remuxed:
            raw_path.unlink(missing_ok=True)
        else:
            shutil.move(str(raw_path), str(out_path))

    report = verify_output(out_path, video_path, sr_ratio or scale)
    report.update({"output_path": str(out_path), "elapsed": time.time() - started, "command": cmd,
                   "config": str(config), "log_path": str(log_path), "model_cls": model_cls(scale),
                   "nproc": nproc, "mode": mode})
    for key in ("size_ok", "audio_ok", "frames_ok"):
        log(f"  {key}: {report[key]}")
    return report


# ------------------------------------------------------------------ CLI


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="valo_upscale.lightx2v_upscale",
                                     description="Upscale video bằng RelayVSR qua LightX2V (2 GPU: Ulysses SP)")
    parser.add_argument("--input")
    parser.add_argument("--out")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--mode", default="sp", choices=list(CONFIG_MODES),
                        help="sp = 2 GPU (Ulysses), single = 1 GPU, block_offload = 1 GPU ít VRAM")
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--gpus", default=None, help="vd 0,1")
    parser.add_argument("--repo-dir", default=str(Path.cwd() / "LightX2V"))
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--token", default=None, help="HF token (mặc định lấy từ HF_TOKEN)")
    parser.add_argument("--weights-source", default=None, help="thư mục đã có weight (vd /kaggle/input)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sr-ratio", type=float, default=None)
    parser.add_argument("--dtype", default="FP16", choices=["FP16", "BF16"])
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--download", action="store_true", help="Tải weight rồi thoát")
    parser.add_argument("--check", action="store_true", help="Kiểm tra môi trường rồi thoát")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in lệnh sẽ chạy")
    args = parser.parse_args(argv)

    log = lambda text: print(text, flush=True)  # noqa: E731
    repo_dir = Path(args.repo_dir)
    model_dir = Path(args.model_dir or repo_dir / "models" / f"RelayVSR_X{args.scale}")
    gpus = [g for g in str(args.gpus).split(",") if g] if args.gpus else None

    if args.download:
        for path in download_weights(args.scale, model_dir, token=args.token,
                                     source_dir=args.weights_source, log=log):
            print("đã có:", path)
        return 0

    if args.check:
        print(json.dumps(check_environment(repo_dir, model_dir, args.scale), ensure_ascii=False, indent=2))
        return 0

    if not (args.input and args.out):
        print("Cần --input và --out (hoặc --check / --download)", file=sys.stderr)
        return 2

    result = run_upscale(args.input, args.out, repo_dir=repo_dir, model_dir=model_dir, scale=args.scale,
                         mode=args.mode, nproc=args.nproc, gpus=gpus, seed=args.seed, sr_ratio=args.sr_ratio,
                         dtype=args.dtype, timeout=args.timeout, progress=log, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in result.items() if k != "command"}, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("LỆNH:", " ".join(str(part) for part in result["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
