"""Upscale video bằng RelayVSR chạy trong ComfyUI (headless, qua HTTP API).

RelayVSR là model **video super-resolution** (Wan 2.2 sparse keyframes + decoder "flash"):
chỉ mỗi **frame thứ 15** đi qua model lớn, các frame còn lại do decoder tái tạo → nhanh hơn
nhiều so với upscale từng frame. Hệ số phóng đại do checkpoint quyết định (x2 hoặc x4).

ComfyUI **core** (master, sau 6/11/2026) đã có sẵn 2 node cho model này:

* ``RelayVSRConditioning(model, images)`` → ``positive`` + ``latent`` (sample **1 step, cfg 1**)
* ``VSRDecode(vae, samples, images, detail)`` → ảnh đã upscale

Graph dùng ở đây (thuần node core, không cần custom node):

    LoadVideo → GetVideoComponents ─┬→ RelayVSRConditioning → KSampler(1 step, cfg 1) → VSRDecode
                                    └→ (images, audio, fps) ─────────────→ CreateVideo → SaveVideo

Audio của video gốc được truyền thẳng vào ``CreateVideo`` nên giữ nguyên tiếng.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import probe

DEFAULT_SERVER = "http://127.0.0.1:8188"
REQUIRED_NODES = (
    "LoadVideo",
    "GetVideoComponents",
    "RelayVSRConditioning",
    "VSRDecode",
    "CreateVideo",
    "SaveVideo",
    "KSampler",
    "UNETLoader",
    "VAELoader",
)

# Cặp model/VAE đi kèm nhau (không được trộn x2 với x4)
MODEL_PAIRS = {
    2: ("relayvsr_x2_lightx2v_int8_convrot.safetensors", "relayvsr_flash_vae_x2_fp32.safetensors"),
    4: ("relayvsr_x4_int8_convrot.safetensors", "relayvsr_flash_vae_x4_fp32.safetensors"),
}
HF_REPO = "Comfy-Org/RelayVSR"


class ComfyError(RuntimeError):
    """Lỗi khi nói chuyện với ComfyUI (thông báo đã rõ cho người dùng)."""


def model_files(scale):
    """Tên file model + VAE cho hệ số 2x/4x (đúng như Comfy-Org đặt tên)."""
    try:
        key = int(scale)
    except (TypeError, ValueError):
        raise ComfyError(f"Hệ số không hợp lệ: {scale!r} (chỉ có 2 hoặc 4).") from None
    if key not in MODEL_PAIRS:
        raise ComfyError(f"RelayVSR chỉ có hệ số 2 hoặc 4, không có {key}x.")
    return MODEL_PAIRS[key]


def normalize_scales(spec):
    """'2' | 4 | '2,4' | [2, 4] → danh sách hệ số cần tải (đã kiểm tra hợp lệ)."""
    if isinstance(spec, (list, tuple, set)):
        items = [str(s) for s in spec]
    else:
        items = [piece.strip() for piece in str(spec).split(",") if piece.strip()]
    scales = []
    for item in items:
        model_files(item)  # validate
        value = int(item)
        if value not in scales:
            scales.append(value)
    if not scales:
        raise ComfyError("Chưa chọn hệ số nào để tải (2, 4 hoặc '2,4').")
    return scales


# ------------------------------------------------------------------ HTTP nhỏ gọn


def _request(base, path, payload=None, timeout=60):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = res.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ComfyError(f"ComfyUI trả HTTP {exc.code} cho {path}: {detail[:600]}") from exc
    except Exception as exc:  # chưa bật server / sai cổng
        raise ComfyError(f"Không gọi được ComfyUI ở {url}: {exc}") from exc
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def log_tail(log_path, lines=12):
    """Đọc mấy dòng cuối log của ComfyUI để báo lỗi cho dễ hiểu."""
    if not log_path:
        return ""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n--- log ComfyUI (cuối) ---\n" + "\n".join(text[-lines:])


def wait_ready(base=DEFAULT_SERVER, timeout=300, interval=2.0, log=None, proc=None, log_path=None):
    """Đợi ComfyUI khởi động xong (server trả /object_info).

    Nếu tiến trình ComfyUI đã thoát (sai cờ, thiếu CUDA, log lỗi...) thì báo ngay kèm log,
    thay vì treo hết thời gian chờ.
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise ComfyError(f"ComfyUI thoát ngay khi khởi động (mã {proc.returncode})."
                             + log_tail(log_path))
        try:
            _request(base, "/object_info", timeout=20)
            return True
        except ComfyError as exc:
            last = exc
            if log:
                log("Đang đợi ComfyUI...")
            time.sleep(interval)
    raise ComfyError(f"ComfyUI không sẵn sàng sau {timeout}s ({last})" + log_tail(log_path))


def object_info(base=DEFAULT_SERVER):
    return _request(base, "/object_info", timeout=60)


def check_ready(base=DEFAULT_SERVER, model_file=None, vae_file=None):
    """Trả dict: node nào thiếu + model/VAE nào chưa có trong ComfyUI."""
    info = object_info(base)
    missing_nodes = [name for name in REQUIRED_NODES if name not in info]
    problems = {}
    if model_file:
        options = info.get("UNETLoader", {}).get("input", {}).get("required", {}).get("unet_name", [[]])[0]
        if model_file not in options:
            problems["model"] = f"{model_file} chưa có trong models/diffusion_models"
    if vae_file:
        options = info.get("VAELoader", {}).get("input", {}).get("required", {}).get("vae_name", [[]])[0]
        if vae_file not in options:
            problems["vae"] = f"{vae_file} chưa có trong models/vae"
    return {"ok": not missing_nodes and not problems, "missing_nodes": missing_nodes, "problems": problems}


def save_video_inputs(format_name="mp4", codec=None, encoding=None):
    """Input cho ``SaveVideo``.

    Lưu ý về ComfyUI: ``format`` là một *DynamicCombo*, giá trị truyền vào prompt phải là
    **key dạng chuỗi** (``"mp4"``), không phải dict — nếu truyền dict thì node sẽ bị gọi
    thiếu tham số ``format``. Các sub-input (``codec``, ``encoding``) cũng là DynamicCombo
    và chỉ cần khi muốn ép codec cụ thể; mặc định để ComfyUI tự chọn (auto → H.264 cho mp4).
    """
    inputs = {"format": format_name}
    if codec:
        inputs["codec"] = codec
    if encoding:
        inputs["encoding"] = encoding
    return inputs


# ------------------------------------------------------------------ workflow


def build_relayvsr_workflow(
    video_file,
    model_file,
    vae_file,
    *,
    steps=1,
    cfg=1.0,
    sampler="euler",
    scheduler="simple",
    seed=0,
    detail=1.0,
    filename_prefix="valo/relayvsr",
    save_format="mp4",
    save_codec=None,
    save_encoding=None,
    create_codec="h264",
    keep_audio=True,
):
    """Workflow dạng API (dict) cho RelayVSR.

    ``video_file`` là tên file **trong thư mục input của ComfyUI**.
    """
    save_inputs = {"video": ["8", 0], "filename_prefix": filename_prefix}
    save_inputs.update(save_video_inputs(save_format, save_codec, save_encoding))
    create_inputs = {"images": ["7", 0], "fps": ["2", 2], "codec": create_codec}
    if keep_audio:
        create_inputs["audio"] = ["2", 1]

    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_file}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": model_file, "weight_dtype": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": vae_file}},
        "5": {"class_type": "RelayVSRConditioning", "inputs": {"model": ["3", 0], "images": ["2", 0]}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["5", 0],
                "negative": ["5", 0],  # cfg = 1.0 nên negative không được dùng
                "latent_image": ["5", 1],
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "7": {
            "class_type": "VSRDecode",
            "inputs": {"vae": ["4", 0], "samples": ["6", 0], "images": ["2", 0], "detail": float(detail)},
        },
        "8": {"class_type": "CreateVideo", "inputs": create_inputs},
        "9": {"class_type": "SaveVideo", "inputs": save_inputs},
    }


def queue_prompt(base, workflow, client_id=None):
    payload = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    result = _request(base, "/prompt", payload, timeout=120)
    if "prompt_id" not in result:
        raise ComfyError(f"ComfyUI không nhận workflow: {json.dumps(result)[:600]}")
    return result["prompt_id"]


def wait_prompt(base, prompt_id, timeout=7200, interval=3.0, on_progress=None, stop_check=None):
    """Đợi workflow chạy xong; trả entry history."""
    started = time.time()
    while time.time() - started < timeout:
        if stop_check and stop_check():
            raise ComfyError("Đã hủy theo yêu cầu.")
        history = _request(base, f"/history/{prompt_id}", timeout=60)
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") in ("success", "error"):
                if status.get("status_str") == "error":
                    raise ComfyError(_error_text(entry))
                return entry
        if on_progress:
            on_progress(int(time.time() - started), _queue_state(base, prompt_id))
        time.sleep(interval)
    raise ComfyError(f"Workflow chạy quá {timeout}s mà chưa xong.")


def _error_text(entry):
    messages = entry.get("status", {}).get("messages") or []
    for name, payload in messages:
        if name == "execution_error":
            node = payload.get("node_type") or payload.get("node_id")
            return f"ComfyUI lỗi ở node {node}: {payload.get('exception_message')} ({payload.get('exception_type')})"
    return f"ComfyUI báo lỗi: {json.dumps(entry.get('status', {}))[:600]}"


def _queue_state(base, prompt_id):
    try:
        queue = _request(base, "/queue", timeout=20)
    except ComfyError:
        return None
    running = queue.get("queue_running") or []
    pending = queue.get("queue_pending") or []
    for index, item in enumerate(running):
        if len(item) > 1 and item[1] == prompt_id:
            return f"đang chạy (vị trí {index + 1})"
    return f"đang chờ ({len(pending)} job trước)" if pending else "đang chạy"


_FOLDER_BY_TYPE = {"output": "output", "input": "input", "temp": "temp"}


def collect_outputs(entry, include_inputs=False):
    """Danh sách file kết quả thật: [{filename, subfolder, type}].

    Lưu ý: ``LoadVideo`` cũng trả về một mục UI cho file *đầu vào* (type ``input``) — phải lọc bỏ,
    nếu không sẽ tưởng nhầm file gốc là kết quả.
    """
    outputs = []
    for node_output in (entry.get("outputs") or {}).values():
        for key in ("images", "video", "videos", "gifs", "files", "audio"):
            for item in node_output.get(key, []) or []:
                if not (isinstance(item, dict) and item.get("filename")):
                    continue
                if (item.get("type") or "output") == "input" and not include_inputs:
                    continue
                outputs.append(item)
    return outputs


def resolve_output_path(comfy_dir, item, output_dir=None):
    """Đường dẫn thật của 1 mục output trong history."""
    folder = _FOLDER_BY_TYPE.get(item.get("type") or "output", "output")
    if folder == "output" and output_dir:
        return Path(output_dir) / (item.get("subfolder") or "") / item["filename"]
    return Path(comfy_dir) / folder / (item.get("subfolder") or "") / item["filename"]


# ------------------------------------------------------------------ chạy 1 video


def _remove_staged(comfy_dir, name, attempts=5, delay=0.6, input_dir=None):
    """Xoá file tạm trong input của ComfyUI (Windows có thể còn giữ handle 1 lúc)."""
    path = Path(input_dir or (Path(comfy_dir) / "input")) / name
    for _ in range(attempts):
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(delay)
    return not path.exists()


def sweep_staged(comfy_dir, prefix="valo_in", min_age=60, input_dir=None):
    """Dọn file tạm của các lần chạy trước (best-effort, không đụng file mới)."""
    removed = 0
    for path in (Path(input_dir or (Path(comfy_dir) / "input"))).glob(f"{prefix}_*"):
        try:
            if time.time() - path.stat().st_mtime > min_age:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def stage_input(comfy_dir, input_path, prefix="valo_in", input_dir=None):
    """Copy video vào thư mục input của ComfyUI, trả tên file để truyền vào LoadVideo."""
    input_dir = Path(input_dir or (Path(comfy_dir) / "input"))
    input_dir.mkdir(parents=True, exist_ok=True)
    sweep_staged(comfy_dir, prefix, input_dir=input_dir)
    target = input_dir / f"{prefix}_{uuid.uuid4().hex[:8]}{Path(input_path).suffix or '.mp4'}"
    shutil.copy2(input_path, target)
    return target.name


def upscale_video(
    input_path,
    output_path,
    *,
    comfy_dir,
    base=DEFAULT_SERVER,
    scale=2,
    model_file=None,
    vae_file=None,
    steps=1,
    cfg=1.0,
    sampler="euler",
    scheduler="simple",
    seed=0,
    detail=1.0,
    timeout=7200,
    progress=None,
    stop_check=None,
    keep_staged=False,
    workflow_builder=None,
    input_dir=None,
    output_dir=None,
):
    """Upscale 1 video qua RelayVSR trong ComfyUI; trả dict kết quả.

    ``workflow_builder`` cho phép thay graph mặc định (dùng khi test hoặc khi bạn muốn
    tự chỉnh workflow), với chữ ký ``(video_file, model_file, vae_file) -> dict``.
    """
    if model_file is None or vae_file is None:
        default_model, default_vae = model_files(scale)
        model_file = model_file or default_model
        vae_file = vae_file or default_vae

    ready = check_ready(base, model_file, vae_file)
    if not ready["ok"]:
        raise ComfyError(
            "ComfyUI chưa sẵn sàng: "
            + "; ".join(ready["missing_nodes"] + list(ready["problems"].values()))
            + ". Cần ComfyUI master sau 06/11/2026 và weight đã tải đúng thư mục."
        )

    staged = stage_input(comfy_dir, input_path, input_dir=input_dir)
    builder = workflow_builder or build_relayvsr_workflow
    if workflow_builder is not None:
        workflow = builder(staged, model_file, vae_file)
    else:
        workflow = builder(
            staged, model_file, vae_file, steps=steps, cfg=cfg, sampler=sampler,
            scheduler=scheduler, seed=seed, detail=detail,
        )
    started = time.time()
    prompt_id = queue_prompt(base, workflow, client_id=uuid.uuid4().hex)
    entry = wait_prompt(base, prompt_id, timeout=timeout, on_progress=progress, stop_check=stop_check)

    outputs = collect_outputs(entry)
    if not outputs:
        raise ComfyError(f"ComfyUI chạy xong nhưng không có file kết quả: {json.dumps(entry)[:400]}")

    first = outputs[0]
    source = resolve_output_path(comfy_dir, first, output_dir=output_dir)
    if not source.exists():
        raise ComfyError(f"Không thấy file kết quả của ComfyUI: {source}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(output_path))

    if not keep_staged:
        _remove_staged(comfy_dir, staged, input_dir=input_dir)

    info = probe.get_media_info(output_path)
    src_info = probe.get_media_info(input_path)
    expected = tuple(dim * scale for dim in (src_info["width"], src_info["height"]))
    return {
        "output_path": str(output_path),
        "scale": scale,
        "model_file": model_file,
        "vae_file": vae_file,
        "info": info,
        "expected_size": expected,
        "size_ok": (info["width"], info["height"]) == expected,
        "elapsed": time.time() - started,
        "outputs": outputs,
    }


# ------------------------------------------------------------------ server


def start_instance(comfy_dir, port=8188, gpu=None, root=None, extra_args=(), log_path=None, env=None,
                   wait=True, timeout=900, log=None):
    """Chạy 1 instance ComfyUI headless, có thể ghim vào 1 GPU và tách thư mục vào/ra riêng.

    Dùng để chạy **nhiều instance song song** (mỗi GPU một instance) mà vẫn **dùng chung thư mục
    ``models``**: mỗi instance có ``--input-directory``/``--output-directory``/``--temp-directory``
    riêng, còn ``--base-directory`` trỏ về cùng thư mục ComfyUI.
    """
    comfy_dir = Path(comfy_dir)
    base_root = Path(root or comfy_dir)
    instance_root = base_root / f"instance_{port}"
    dirs = {
        "input_dir": instance_root / "input",
        "output_dir": instance_root / "output",
        "temp_dir": instance_root / "temp",
        "user_dir": instance_root / "user",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    # input/output/temp/user riêng cho từng instance; models dùng chung qua --base-directory
    # (user-dir riêng để 2 instance không tranh nhau database SQLite)
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port),
           "--disable-auto-launch", "--base-directory", str(comfy_dir),
           "--input-directory", str(dirs["input_dir"]), "--output-directory", str(dirs["output_dir"]),
           "--temp-directory", str(dirs["temp_dir"]), "--user-directory", str(dirs["user_dir"])]
    cmd += [str(a) for a in extra_args]

    run_env = dict(env or os.environ)
    if gpu is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "wb") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, cwd=str(comfy_dir), stdout=handle, stderr=subprocess.STDOUT, env=run_env)
    proc._valo_log = handle  # giữ tham chiếu để không bị đóng

    try:
        instance = ComfyInstance(comfy_dir=str(comfy_dir), base=f"http://127.0.0.1:{port}", proc=proc,
                                 gpu=gpu, **{k: str(v) for k, v in dirs.items()})
        if wait:
            wait_ready(instance.base, timeout=timeout, log=log, proc=proc, log_path=log_path)
    except BaseException:  # không để lại tiến trình mồ côi
        stop_server(proc)
        raise
    return instance


def start_server(comfy_dir, port=8188, extra_args=(), log_path=None, env=None):
    """Chạy ComfyUI headless (không tự mở trình duyệt). Trả Popen (API cũ)."""
    instance = start_instance(comfy_dir, port=port, extra_args=extra_args, log_path=log_path, env=env, wait=False)
    return instance.proc


def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@dataclass
class ComfyInstance:
    """Một instance ComfyUI (thường là 1 GPU) với thư mục vào/ra riêng."""

    comfy_dir: str
    base: str
    input_dir: str = None
    output_dir: str = None
    temp_dir: str = None
    user_dir: str = None
    gpu: object = None
    proc: object = None
    results: list = field(default_factory=list)

    def upscale(self, input_path, output_path, **kwargs):
        return upscale_video(input_path, output_path, comfy_dir=self.comfy_dir, base=self.base,
                             input_dir=self.input_dir, output_dir=self.output_dir, **kwargs)

    def stop(self):
        stop_server(self.proc)


def start_instances(comfy_dir, ports=None, gpus=None, root=None, extra_args=(), log_dir=None, log=None):
    """Bật nhiều instance ComfyUI (mỗi GPU một instance), dùng chung thư mục models.

    Trả danh sách :class:`ComfyInstance`. Nhớ gọi ``stop()``/``stop_instances()`` khi xong
    (hoặc dùng :func:`run_clips_parallel` để tự dọn).
    """
    if gpus is None:
        try:
            import torch  # noqa: PLC0415

            count = torch.cuda.device_count()
        except Exception:
            count = 0
        gpus = list(range(count)) if count > 0 else [None]
    elif isinstance(gpus, (int, str)) or gpus is None:
        gpus = [gpus]
    else:
        gpus = list(gpus)

    ports = list(ports) if ports else [8188 + index for index in range(len(gpus))]
    if len(ports) != len(gpus):
        raise ComfyError(f"Số port ({len(ports)}) phải bằng số GPU ({len(gpus)}).")

    instances = []
    try:
        for gpu, port in zip(gpus, ports):
            if log:
                log(f"Khởi động ComfyUI ở port {port}" + (f" (GPU {gpu})" if gpu is not None else ""))
            instances.append(start_instance(
                comfy_dir, port=port, gpu=gpu, root=root, extra_args=extra_args,
                log_path=str(Path(log_dir) / f"comfy_{port}.log") if log_dir else None, log=log,
            ))
    except Exception:
        stop_instances(instances)
        raise
    return instances


def stop_instances(instances):
    for instance in instances or []:
        try:
            instance.stop()
        except Exception:
            pass


def run_clips_parallel(clips, out_dir, *, comfy_dir, gpus=None, ports=None, scale=2, model_file=None,
                       vae_file=None, root=None, log_dir=None, extra_args=(), on_log=None, stop_check=None,
                       **upscale_kwargs):
    """Chạy nhiều clip **song song**: mỗi GPU một instance ComfyUI, chia clip theo vòng.

    Lưu ý: cách này tăng **thông lượng khi có nhiều clip** (2 GPU ≈ 2× số clip mỗi giờ),
    không làm 1 clip chạy nhanh hơn. Muốn 1 clip dùng 2 GPU thì phải dùng LightX2V
    với config ``relayvsr_x2_sp.json`` (xem README).
    """
    clips = [Path(c) for c in clips]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = on_log or (lambda text: None)
    if not clips:
        return {"results": [], "errors": [], "instances": 0, "clips": [], "wall": 0.0}

    started = time.time()
    instances = start_instances(comfy_dir, ports=ports, gpus=gpus, root=root, log_dir=log_dir,
                                extra_args=extra_args, log=log)
    results, errors = [None] * len(clips), []
    try:
        with ThreadPoolExecutor(max_workers=len(instances)) as pool:
            futures = {}
            for index, clip in enumerate(clips):
                instance = instances[index % len(instances)]
                scale_for_clip = int(scale)
                output = out_dir / f"{clip.stem}_relayvsr_x{scale_for_clip}.mp4"
                log(f"→ [{index + 1}/{len(clips)}] {clip.name} (port {instance.base.rsplit(':', 1)[-1]})")
                futures[pool.submit(instance.upscale, clip, output, scale=scale_for_clip, model_file=model_file,
                                    vae_file=vae_file, stop_check=stop_check, **upscale_kwargs)] = index
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    results[index] = result
                    log(f"✔ [{index + 1}/{len(clips)}] {clips[index].name} — "
                        f"{result['info']['width']}x{result['info']['height']} • {result['elapsed']:.1f}s")
                except Exception as exc:  # noqa: BLE001 - báo hết rồi trả về
                    errors.append((clips[index].name, str(exc)))
                    log(f"✗ [{index + 1}/{len(clips)}] {clips[index].name}: {exc}")
    finally:
        stop_instances(instances)
    return {"results": results, "errors": errors, "instances": len(instances),
            "clips": [str(c) for c in clips], "wall": time.time() - started}


def _hf_token(token=None):
    """Token lấy theo thứ tự: tham số → HF_TOKEN → HUGGING_FACE_HUB_TOKEN."""
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def find_local_weight(source_dir, filename):
    """Tìm file weight đã có sẵn (ví dụ trong /kaggle/input của một Kaggle Dataset)."""
    if not source_dir:
        return None
    root = Path(source_dir)
    if not root.exists():
        return None
    for path in root.rglob(filename):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def link_or_copy(src, dst):
    """Symlink nếu được (không tốn thêm đĩa), không thì copy."""
    dst = Path(dst)
    try:
        dst.symlink_to(Path(src))
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def download_models(comfy_dir, scale=2, repo=HF_REPO, token=None, source_dir=None, log=None):
    """Đưa model + VAE vào đúng thư mục ComfyUI.

    * ``scale``: ``2``, ``4`` hoặc ``"2,4"`` (tải cả hai cặp — mỗi cặp ~5,3 GB).
    * ``token``: Hugging Face token (mặc định lấy từ biến môi trường ``HF_TOKEN``); repo này
      công khai nên token chỉ để tăng hạn mức và tốc độ tải.
    * ``source_dir``: nếu đã có weight ở đâu đó (thường là ``/kaggle/input`` của một Kaggle
      Dataset) thì symlink/copy từ đó, **không tải lại** mỗi phiên.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ComfyError("Thiếu huggingface_hub. Cài: pip install -q huggingface_hub") from exc

    scales = normalize_scales(scale)
    token = _hf_token(token)
    models_dir = Path(comfy_dir) / "models"

    plan = []
    for item in scales:
        model_file, vae_file = model_files(item)
        plan += [
            (f"diffusion_models/{model_file}", models_dir / "diffusion_models" / model_file),
            (f"vae/{vae_file}", models_dir / "vae" / vae_file),
        ]

    paths = []
    for remote, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0:
            if log:
                log(f"đã có sẵn: {target}")
            paths.append(str(target))
            continue

        local = find_local_weight(source_dir, Path(remote).name)
        if local is not None:
            how = link_or_copy(local, target)
            if log:
                log(f"{how} từ {local} → {target}")
            paths.append(str(target))
            continue

        if log:
            log(f"Đang tải {remote} {'(đã xác thực HF_TOKEN)' if token else '(không xác thực)'} ...")
        paths.append(
            hf_hub_download(repo_id=repo, filename=remote, local_dir=str(models_dir), token=token)
        )
    return paths


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="valo_upscale.comfy_upscale",
                                     description="Upscale video bằng RelayVSR trong ComfyUI (headless)")
    parser.add_argument("--input", help="Video đầu vào")
    parser.add_argument("--out", help="File kết quả")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4], help="Hệ số dùng khi chạy")
    parser.add_argument("--scales", default=None,
                        help="Hệ số cần TẢI khi --download: \"2\", \"4\" hoặc \"2,4\" (mặc định = --scale)")
    parser.add_argument("--comfy-dir", help="Thư mục ComfyUI (có main.py)")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--detail", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra node/weight rồi thoát")
    parser.add_argument("--download", action="store_true", help="Tải weight về ComfyUI rồi thoát")
    parser.add_argument("--token", default=None, help="Hugging Face token (mặc định lấy từ HF_TOKEN)")
    parser.add_argument("--weights-source", default=None,
                        help="Thư mục đã có weight (vd /kaggle/input) — symlink thay vì tải lại")
    args = parser.parse_args(argv)

    log = lambda text: print(text, flush=True)  # noqa: E731

    if args.download:
        if not args.comfy_dir:
            print("Cần --comfy-dir", file=sys.stderr)
            return 2
        for path in download_models(args.comfy_dir, args.scales or args.scale, token=args.token,
                                   source_dir=args.weights_source, log=log):
            print("đã có:", path)
        return 0

    if args.check:
        model_file, vae_file = model_files(args.scale)
        state = check_ready(args.server, model_file, vae_file)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if state["ok"] else 1

    if not (args.input and args.out and args.comfy_dir):
        print("Cần --input, --out và --comfy-dir", file=sys.stderr)
        return 2

    result = upscale_video(
        args.input, args.out, comfy_dir=args.comfy_dir, base=args.server, scale=args.scale,
        steps=args.steps, cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler,
        detail=args.detail, seed=args.seed, timeout=args.timeout, progress=lambda s, st: log(f"  [{s}s] {st}"),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "outputs"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
