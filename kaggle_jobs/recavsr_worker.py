"""Worker chạy trên **Kaggle GPU**: ComfyUI + ReCaVSR (int8) để upscale video.

Được cả hai phía dùng:

* **Ứng dụng Windows** (`winapp/`): app chép file này vào thư mục kernel, thay dòng ``JOB = …`` bằng
  tham số người dùng chọn (địa chỉ tải video + tuỳ chọn), rồi ``kaggle kernels push``.
* **Chạy tay** (để gỡ lỗi hoặc dùng không cần app):

      python kaggle_jobs/recavsr_worker.py --video clip.mp4 --scale 4 --precision int8

Kết quả ghi vào ``/kaggle/working`` (tải về bằng ``kaggle kernels output`` hoặc tab Output của Kaggle).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

# >>>JOB_JSON — ứng dụng Windows thay dòng dưới bằng tham số thật
JOB = {}
# <<<JOB_JSON

VALO_REPO = "https://github.com/hoannguyen171/UPSCALEVIDEO.git"
RECAVSR_NODE_REPO = "https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR"

WORK = Path(os.environ.get("VALO_WORK", "/kaggle/working" if Path("/kaggle/working").is_dir() else "."))
_LOG_HANDLE = None


def log(*parts):
    text = " ".join(str(p) for p in parts)
    line = f"[{time.strftime('%H:%M:%S')}] {text}"
    print(line, flush=True)
    if _LOG_HANDLE:
        _LOG_HANDLE.write(line + "\n")
        _LOG_HANDLE.flush()


def run(cmd, cwd=None, check=False, env=None, timeout=None):
    log("$", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, timeout=timeout,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    tail = (result.stdout or "").strip().splitlines()[-3:]
    for line in tail:
        log("   |", line[:200])
    if check and result.returncode != 0:
        raise RuntimeError(f"lệnh thất bại ({result.returncode}): {' '.join(str(c) for c in cmd)}")
    return result.returncode


def pip_install(*packages, optional=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages, "--disable-pip-version-check"]
    code = run(cmd)
    if code != 0 and not optional:
        raise RuntimeError(f"pip install thất bại: {packages}")
    return code == 0


def fetch_video(url, dest_dir, name=None, attempts=3, delay=20):
    """Tải video từ URL công khai (app Windows chia sẻ qua tunnel) — thử lại vài lần.

    Gọi **trước** khi cài ComfyUI/tải weight: tunnel của app chỉ mở trong lúc job chạy, tải sớm
    thì không phụ thuộc việc cài đặt mất bao lâu.
    """
    last = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _fetch_video_once(url, dest_dir, name)
        except Exception as exc:  # noqa: BLE001 - thử lại rồi mới báo
            last = exc
            log(f"tải video lần {attempt}/{attempts} lỗi: {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(delay)
    raise RuntimeError(f"Không tải được video từ URL chia sẻ sau {attempts} lần: {last}")


def _fetch_video_once(url, dest_dir, name=None):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = name or Path(urllib.parse.urlparse(url).path).name or "input.mp4"
    target = dest_dir / name
    log(f"Đang tải video từ {url.split('?')[0][:80]} ...")
    with urllib.request.urlopen(url, timeout=300) as response, open(target, "wb") as handle:
        total = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            if total % (50 << 20) < (1 << 20):
                log(f"   đã tải {total / 1048576:.0f} MB")
    log(f"đã tải {target.name} ({target.stat().st_size / 1048576:.1f} MB)")
    return target


def prepare_comfy(comfy_dir, log_dir):
    """Cài ComfyUI master + node pack ReCaVSR."""
    comfy_dir = Path(comfy_dir)
    if not (comfy_dir / "main.py").exists():
        run(["git", "clone", "--depth", "1", "-q", "https://github.com/comfyanonymous/ComfyUI.git", str(comfy_dir)],
            check=True)
    else:
        run(["git", "-C", str(comfy_dir), "pull", "-q", "--ff-only"])

    requirements = comfy_dir / "requirements.txt"
    skip = ("torch", "torchvision", "torchaudio")
    filtered = comfy_dir / "requirements-notorch.txt"          # giữ nguyên torch CUDA của Kaggle
    filtered.write_text("\n".join(line for line in requirements.read_text(encoding="utf-8").splitlines()
                                  if line.split("==")[0].strip() not in skip), encoding="utf-8")
    pip_install("-r", str(filtered))
    # BẮT BUỘC: node pack ReCaVSR import `tl` từ triton; thiếu triton là không nạp được node
    # (flashdecoder.py `from .attention import rotate_pairs, triton, tl`) → cài chắc chắn, báo rõ nếu lỗi
    if not pip_install("triton", "--disable-pip-version-check"):
        raise RuntimeError("Không cài được triton — node ReCaVSR cần triton để import (xem README của node pack)")
    pip_install("opencv-python-headless", "huggingface_hub", optional=True)

    node_dir = comfy_dir / "custom_nodes" / "ComfyUI-ReCaVSR"
    if not (node_dir / ".git").exists():
        run(["git", "clone", "--depth", "1", "-q", RECAVSR_NODE_REPO, str(node_dir)], check=True)
    else:
        run(["git", "-C", str(node_dir), "pull", "-q", "--ff-only"])
    node_requirements = node_dir / "requirements.txt"
    if node_requirements.exists():
        pip_install("-r", str(node_requirements), optional=True)
    return comfy_dir, node_dir


def install_valo_repo(repo_dir):
    """Clone repo này để dùng `valo_upscale` (client ComfyUI + module ReCaVSR)."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / ".git").exists():
        run(["git", "clone", "--depth", "1", "-q", VALO_REPO, str(repo_dir)], check=True)
    sys.path.insert(0, str(repo_dir))
    return repo_dir


def main(argv=None):
    global _LOG_HANDLE
    parser = argparse.ArgumentParser(description="Worker ReCaVSR trên Kaggle (ComfyUI + int8)")
    parser.add_argument("--video", default=JOB.get("video"), help="Đường dẫn video local (khi chạy tay)")
    parser.add_argument("--video-url", default=JOB.get("video_url"), help="URL tải video (app Windows dùng)")
    parser.add_argument("--scale", type=float, default=JOB.get("scale", 4.0))
    parser.add_argument("--precision", default=JOB.get("precision", "int8"), choices=["int8", "bf16"])
    parser.add_argument("--color-fix", default=JOB.get("color_fix", "none"), choices=["none", "adain", "wavelet"])
    parser.add_argument("--kernels", default=JOB.get("kernels", "fast"), choices=["fast", "exact"])
    parser.add_argument("--encoder", default=JOB.get("encoder", "auto"))
    parser.add_argument("--seed", type=int, default=int(JOB.get("seed", 42)))
    parser.add_argument("--kv-offload", action="store_true", default=bool(JOB.get("kv_offload", False)))
    parser.add_argument("--lowvram", action="store_true", default=bool(JOB.get("lowvram", True)))
    parser.add_argument("--out-name", default=JOB.get("output_name"))
    parser.add_argument("--work-dir", default=str(WORK))
    parser.add_argument("--keep-comfy", action="store_true", default=bool(JOB.get("keep_comfy", False)),
                        help="Giữ ComfyUI + weight trong /kaggle/working (chỉ để gỡ lỗi)")
    args = parser.parse_args(argv)

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    _LOG_HANDLE = open(work / "valo_run.log", "a", encoding="utf-8")
    started = time.time()
    summary = {"ok": False}

    try:
        log("=== Valo ReCaVSR worker ===")
        log("python", sys.version.split()[0], "| work", work)
        import torch  # noqa: PLC0415

        log("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| GPU:", torch.cuda.device_count())
        if not torch.cuda.is_available():
            raise RuntimeError("Không thấy GPU — bật Accelerator = GPU T4 x2 cho notebook này")

        repo_dir = install_valo_repo(work / "valo-upscale-gpu")
        from valo_upscale import recavsr  # noqa: PLC0415
        from valo_upscale import comfy_upscale as cu  # noqa: PLC0415

        # 1) lấy video TRƯỚC (rẻ, và tunnel của app chỉ mở trong lúc job chạy)
        if args.video_url:
            video = fetch_video(args.video_url, work / "input")
        elif args.video:
            video = Path(args.video)
            if not video.exists():
                raise FileNotFoundError(f"không thấy video {video}")
        else:
            found = sorted(Path("/kaggle/input").rglob("*.mp4"))
            if not found:
                raise FileNotFoundError("Không có --video-url/--video và /kaggle/input cũng không có .mp4")
            video = found[0]
            log("dùng video đầu tiên trong /kaggle/input:", video)

        # 2) cài ComfyUI + node pack (chậm), rồi tải weight
        comfy_dir, node_dir = prepare_comfy(work / "ComfyUI", work / "logs")
        log("ComfyUI:", comfy_dir, "| node pack:", node_dir)

        models_dir = comfy_dir / "models" / "recavsr"
        recavsr.download_weights(models_dir, args.precision, token=os.environ.get("HF_TOKEN"),
                                 source_dir="/kaggle/input" if Path("/kaggle/input").is_dir() else None, log=log)
        log("weight:", [p.name for p in models_dir.glob("*.safetensors")])

        extra_args = ("--lowvram",) if args.lowvram else ()
        instance = cu.start_instance(comfy_dir, port=8188, extra_args=extra_args,
                                     log_path=str(work / "comfy.log"), log=lambda t: None,
                                     env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        try:
            state = recavsr.check_ready(instance.base, recavsr.transformer_name(args.precision))
            log("check_ready:", state)
            if not state["ok"]:
                raise RuntimeError(f"ReCaVSR chưa sẵn sàng: {state}")

            out_name = args.out_name or f"{Path(video).stem}_recavsr_x{args.scale:g}.mp4"
            out_path = work / out_name
            result = recavsr.upscale_video(
                video, out_path, comfy_dir=str(comfy_dir), base=instance.base, scale=args.scale,
                precision=args.precision, color_fix=args.color_fix, kernels=args.kernels,
                encoder=args.encoder, seed=args.seed, kv_offload=args.kv_offload, timeout=10800,
                progress=lambda percent, stage, text: log(f"  [{percent:3d}%] {stage}: {text}"),
            )
            summary = {
                "ok": Path(result["output_path"]).exists(),
                "output_path": result["output_path"],
                "output_name": Path(result["output_path"]).name,
                "size": f"{result['info']['width']}x{result['info']['height']}",
                "frames": result["info"]["nb_frames"],
                "fps": result["info"]["fps"],
                "has_audio": result["info"]["has_audio"],
                "size_ok": result["size_ok"],
                "elapsed": round(result["elapsed"], 1),
                "precision": args.precision,
                "scale": args.scale,
                "kernels": args.kernels,
            }
            log("KẾT QUẢ:", json.dumps(summary, ensure_ascii=False))
        finally:
            cu.stop_instances([instance])
    except Exception as exc:  # noqa: BLE001 - in ra để app đọc được lý do
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log("LỖI:", summary["error"])
        log(traceback.format_exc()[-1500:])
    finally:
        if not args.keep_comfy:
            # /kaggle/working là thứ `kaggle kernels output` tải về: chỉ giữ video kết quả + json + log
            for junk in (work / "ComfyUI", work / "input"):
                try:
                    shutil.rmtree(junk, ignore_errors=True)
                except OSError:
                    pass
            log("đã dọn ComfyUI + weight khỏi /kaggle/working (chỉ giữ kết quả)")
        summary["total_elapsed"] = round(time.time() - started, 1)
        (work / "valo_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log("=== XONG === ", json.dumps(summary, ensure_ascii=False))
        if _LOG_HANDLE:
            _LOG_HANDLE.close()
        if not summary.get("ok"):
            print("VALO_FAILED", flush=True)
            return 1
        print("VALO_OK", flush=True)
        return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  (dùng trong fetch_video)

    raise SystemExit(main())
