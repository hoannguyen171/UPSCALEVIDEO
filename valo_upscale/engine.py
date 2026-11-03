"""Engine upscale — bản port Python của src/upscaler.js trong app gốc.

Hai engine giống app:

1. ``realesrgan`` (AI, GPU CUDA): suy luận Real-ESRGAN bằng PyTorch, ảnh native 4x rồi resample
   Lanczos (+unsharp) cho các mốc 2x/3x/6x/8x, giống cách app dùng ncnn native scale rồi ffmpeg tinh chỉnh.
2. ``ffmpeg_hq`` (CPU, không cần GPU): scale Lanczos + unsharp, video có chuỗi 3 pass tự cứu.

Tối ưu tốc độ (quan trọng khi chạy Kaggle T4 x2):

* **Nhiều GPU**: mỗi GPU một model riêng, chia việc theo patch (ảnh) hoặc theo khối frame (video).
* **Batch theo patch/frame**: gộp nhiều patch vào 1 lần forward → GPU không bị "đói" (util thấp).
* **Chuyển dữ liệu rẻ**: đẩy uint8 lên GPU rồi mới đổi sang half/float và chia 255; trả về uint8.
* **channels_last + cudnn.benchmark**: tăng tốc conv trên T4.
* **Pipeline đa luồng**: đọc/giải mã frame (PIL) và ghi frame chạy song song với suy luận.
* **NVENC/NVDEC**: giải mã và encode video bằng phần cứng, đỡ nghẽn CPU (tự dò, tự fallback libx264).
* **Cache model**: nạp model 1 lần cho cả thư mục (không nạp lại mỗi file).
* **Tự hạ batch / tự bật tile khi hết VRAM** thay vì báo lỗi.
"""

import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import fftools, models, probe
from .dims import calculate_image_dimensions, calculate_video_dimensions

IMAGE_UNSHARP = dict(radius=3, percent=30, threshold=3)
VIDEO_FRAME_QUALITY = {"jpg": ["-q:v", "1"], "png": ["-pix_fmt", "rgb24"]}
DEFAULT_BATCH = 4
OOM_TILE_FALLBACK = 400


class UpscaleCancelled(RuntimeError):
    pass


def _emit(progress, percent, stage, text):
    if progress:
        progress(int(percent), stage, text)


def _check_stop(stop_check):
    if stop_check and stop_check():
        raise UpscaleCancelled("Đã hủy theo yêu cầu.")


def _numpy_to_pil(arr):
    return Image.fromarray(np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8)), mode="RGB")


def parse_devices(spec=None):
    """'auto' | 'cpu' | '0' | '0,1' | ['cuda:0'] -> list thiết bị torch."""
    import torch

    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "auto", "all")):
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return ["mps"]
        return ["cpu"]

    if isinstance(spec, (list, tuple)):
        items = [str(s) for s in spec]
    else:
        items = [s.strip() for s in str(spec).split(",") if s.strip()]

    devices = []
    for item in items:
        low = item.lower()
        if low == "cpu":
            devices.append("cpu")
        elif low.startswith("cuda"):
            devices.append(item)
        else:
            devices.append(f"cuda:{int(item)}")
    return devices or ["cpu"]


class _DeviceWorker:
    """Một thiết bị (GPU/CPU) với model riêng, tự batch hoá và tự hạ tải khi hết VRAM."""

    def __init__(self, device, model_name, tile=0, tile_pad=10, fp32=False, batch=DEFAULT_BATCH):
        import torch

        self.device = device
        self.model_name = model_name
        self.tile = int(tile or 0)
        self.tile_pad = int(tile_pad or 10)
        self.batch = max(1, int(batch or 1))
        self.is_cuda = device.startswith("cuda")
        self.half = self.is_cuda and not fp32

        torch.backends.cudnn.benchmark = True
        self.net, self.native = models.load_model(model_name, device=device, half=self.half)
        self.divisor = getattr(self.net, "divisor", 1)
        self.out_scale = getattr(self.net, "out_scale", self.native)
        if self.native != self.out_scale:
            raise RuntimeError(f"Hệ số native của {model_name} không nhất quán ({self.native} vs {self.out_scale}).")
        self.channels_last = self.is_cuda
        if self.channels_last:
            try:
                self.net = self.net.to(memory_format=torch.channels_last)
            except Exception:
                self.channels_last = False

    # ------------------------------------------------------------- hình học patch
    def plan(self, height, width):
        """Kế hoạch cắt patch cố định kích thước để batch được."""
        tile = self.tile
        if tile <= 0 or (height <= tile and width <= tile):
            return {
                "tile": False,
                "windows": [(0, 0, height, width)],
                "pad": 0,
                "size": 0,
                "patch_shape": (height, width, 3),
            }

        pad = self.tile_pad
        pad_b = (tile - height % tile) % tile
        pad_r = (tile - width % tile) % tile
        windows = [
            (y, x, min(tile, height - y), min(tile, width - x))
            for y in range(0, height, tile)
            for x in range(0, width, tile)
        ]
        return {
            "tile": True,
            "windows": windows,
            "pad": pad,
            "pad_b": pad_b,
            "pad_r": pad_r,
            "size": tile,
            "patch_shape": (tile + 2 * pad, tile + 2 * pad, 3),
        }

    def patches_of(self, img, plan):
        if not plan["tile"]:
            return [np.ascontiguousarray(img)]
        pad, pad_b, pad_r = plan["pad"], plan["pad_b"], plan["pad_r"]
        padded = np.pad(img, ((pad, pad + pad_b), (pad, pad + pad_r), (0, 0)), mode="edge")
        side = plan["size"] + 2 * pad
        return [
            np.ascontiguousarray(padded[y : y + side, x : x + side])
            for y, x, _, _ in plan["windows"]
        ]

    def rebuild(self, patches, plan, shape):
        import torch  # noqa: F401  (giữ import cục bộ cho rõ ràng luồng GPU)

        height, width = shape
        scale = self.out_scale
        if not plan["tile"]:
            return np.ascontiguousarray(patches[0][: height * scale, : width * scale])

        canvas = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
        pad = plan["pad"]
        for (y, x, ch, cw), patch in zip(plan["windows"], patches):
            core = patch[pad * scale : pad * scale + ch * scale, pad * scale : pad * scale + cw * scale]
            canvas[y * scale : y * scale + ch * scale, x * scale : x * scale + cw * scale] = core
        return canvas

    # ------------------------------------------------------------------ suy luận
    def run(self, patches):
        """Batch nhiều patch cùng kích thước -> list ảnh uint8."""
        batch = self.batch
        results = []
        index = 0
        while index < len(patches):
            chunk = patches[index : index + batch]
            try:
                results.extend(self._forward(chunk))
                index += len(chunk)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self._free_cuda()
                if batch > 1:
                    batch = max(1, batch // 2)
                    self.batch = min(self.batch, batch)
                    continue
                raise
        return results

    def _forward(self, patches):
        import torch

        if self.divisor > 1:  # pixel_unshuffle (RRDBNet 2x/1x) cần cạnh chia hết cho divisor
            fixed = []
            for patch in patches:
                h, w = patch.shape[:2]
                pad_h = (self.divisor - h % self.divisor) % self.divisor
                pad_w = (self.divisor - w % self.divisor) % self.divisor
                fixed.append(
                    np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") if (pad_h or pad_w) else patch
                )
            patches = fixed

        stack = np.stack(patches)  # uint8 (B,H,W,3) — truyền lên GPU ở dạng uint8 cho nhẹ
        with torch.no_grad():
            tensor = torch.from_numpy(stack).to(self.device, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2)
            if self.channels_last:
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            else:
                tensor = tensor.contiguous()
            if self.half:
                tensor = tensor.half()
            out = self.net(tensor.div(255.0))
            out = out.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            out = out.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        return [
            out[i][: patches[i].shape[0] * self.out_scale, : patches[i].shape[1] * self.out_scale]
            for i in range(len(patches))
        ]

    def _free_cuda(self):
        if self.is_cuda:
            import torch

            torch.cuda.empty_cache()

    def upscale_images(self, images):
        """Plan -> patch -> forward -> ghép lại, chỉ dùng thiết bị này (dùng cho 1 luồng video)."""
        output = [None] * len(images)
        for index, img in enumerate(images):
            plan = self.plan(*img.shape[:2])
            patches = self.patches_of(img, plan)
            output[index] = self.rebuild(self.run(patches), plan, img.shape[:2])
        return output

    def close(self):
        self.net = None
        if self.is_cuda:
            self._free_cuda()


class RealESRGANEngine:
    """Suy luận Real-ESRGAN trên PyTorch, tận dụng mọi GPU có sẵn."""

    def __init__(self, model=None, tile=0, tile_pad=10, fp32=False, device=None, devices=None,
                 batch=DEFAULT_BATCH, encoder="auto", hwaccel=True, io_threads=2):
        self.model_name = models.resolve_model_name(model or models.DEFAULT_IMAGE_MODEL)
        self.tile = int(tile or 0)
        self.tile_pad = int(tile_pad or 10)
        self.fp32 = bool(fp32)
        self.batch = max(1, int(batch or 1))
        self.devices = parse_devices(devices if devices is not None else device)
        self.encoder = encoder
        self.hwaccel = bool(hwaccel)
        self.io_threads = max(1, int(io_threads or 1))
        self._workers = {}
        self._lock = threading.Lock()
        self._pool = None

    # ---------------------------------------------------------------- thiết bị
    @property
    def device(self):
        return self.devices[0]

    @property
    def half(self):
        return self.devices[0].startswith("cuda") and not self.fp32

    @property
    def effective_batch(self):
        """Batch thực tế: chỉ áp dụng trên GPU (CPU bị ép về 1)."""
        return self.batch if self.devices[0].startswith("cuda") else 1

    def use_model(self, model_name=None):
        key = models.resolve_model_name(model_name or self.model_name)
        self.model_name = key
        return key

    def _worker(self, device, model_name=None):
        key = self.use_model(model_name)
        # Batch chỉ có lợi trên GPU (tăng mức sử dụng SM). Trên CPU, batch lớn làm tràn cache
        # và chậm hơn, nên ép về 1.
        batch = self.batch if device.startswith("cuda") else 1
        with self._lock:
            worker = self._workers.get(device)
            if worker is None or worker.model_name != key:
                worker = _DeviceWorker(device, key, tile=self.tile, tile_pad=self.tile_pad,
                                       fp32=self.fp32, batch=batch)
                self._workers[device] = worker
            else:
                worker.batch = min(worker.batch, batch)
        return worker

    @property
    def workers(self):
        return [self._worker(device) for device in self.devices]

    def net(self, model_name=None):
        """Tương thích API cũ: trả (net, native) của thiết bị đầu tiên."""
        worker = self._worker(self.device, model_name)
        return worker.net, worker.native

    def close(self):
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None

    def _executor(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.devices)), thread_name_prefix="valo-infer")
        return self._pool

    # ---------------------------------------------------------------- suy luận
    def _reduce_for_oom(self):
        changed = False
        for worker in self.workers:
            if worker.batch > 1:
                worker.batch = max(1, worker.batch // 2)
                changed = True
            elif worker.tile <= 0:
                worker.tile = OOM_TILE_FALLBACK
                changed = True
            elif worker.tile > 64:
                worker.tile = max(64, worker.tile // 2)
                changed = True
            worker._free_cuda()
        return changed

    def upscale_arrays(self, images, model_name=None, worker=None):
        """images: list uint8 HxWx3 -> list uint8 (native x). Tự chia nhiều GPU."""
        if not images:
            return []
        model_key = self.use_model(model_name)

        for _ in range(6):
            try:
                if worker is not None:
                    return worker.upscale_images(images)
                return self._upscale_split(images, model_key)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                if not self._reduce_for_oom():
                    raise
        raise RuntimeError("Hết VRAM ngay cả khi đã giảm batch và tile.")

    def _upscale_split(self, images, model_key):
        """Chia patch của nhiều ảnh cho tất cả GPU rồi ghép lại."""
        workers = [self._worker(device, model_key) for device in self.devices]
        results = [None] * len(images)

        # gom theo kích thước để mọi patch trong 1 lần forward cùng shape
        groups = {}
        for index, img in enumerate(images):
            groups.setdefault(img.shape[:2], []).append(index)

        for shape, indexes in groups.items():
            plans = [workers[0].plan(*shape) for _ in indexes]
            per_image = [workers[0].patches_of(images[i], plans[k]) for k, i in enumerate(indexes)]
            flat = [patch for patches in per_image for patch in patches]

            if len(workers) == 1 or len(flat) < 2:
                outs = workers[0].run(flat)
            else:
                bounds = [round(len(flat) * i / len(workers)) for i in range(len(workers) + 1)]
                futures = [
                    self._executor().submit(w.run, flat[bounds[i] : bounds[i + 1]])
                    for i, w in enumerate(workers)
                    if bounds[i + 1] > bounds[i]
                ]
                outs = [item for future in futures for item in future.result()]

            count = len(plans[0]["windows"])
            for k, index in enumerate(indexes):
                results[index] = workers[0].rebuild(outs[k * count : (k + 1) * count], plans[k], shape)
        return results

    # giữ API cũ cho test/notebook: float32 [0,1] vào, float32 ra
    def infer(self, img, model_name=None):
        array = np.clip(np.asarray(img, dtype=np.float32) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out = self.upscale_arrays([array], model_name=model_name)[0]
        return out.astype(np.float32) / 255.0

    # ------------------------------------------------------------------ ảnh
    def upscale_image(self, input_path, output_path, scale, model_name=None, progress=None, stop_check=None):
        src = Image.open(input_path)
        alpha = None
        if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
            alpha = src.convert("RGBA").getchannel("A")
        width, height = src.size

        key = self.use_model(model_name)
        native = models.MODEL_REGISTRY[key]["native"]
        _emit(progress, 5, "upscaling",
              f"AI Upscale {scale}x ({key}, native {native}x, {self.devices})...")

        started = time.time()
        arr = np.asarray(src.convert("RGB"), dtype=np.uint8)
        out = self.upscale_arrays([arr], model_name=key)[0]
        _check_stop(stop_check)

        tw, th = calculate_image_dimensions(width, height, scale)
        result = _numpy_to_pil(out)
        if result.size != (tw, th):
            _emit(progress, 88, "refining", f"Tinh chỉnh độ nét về {tw}x{th} ({scale}x)...")
            result = result.resize((tw, th), Image.LANCZOS)
            if scale > native:
                result = result.filter(ImageFilter.UnsharpMask(**IMAGE_UNSHARP))
        if alpha is not None:
            result = result.convert("RGB")
            result.putalpha(alpha.resize((tw, th), Image.LANCZOS))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path, "PNG" if alpha is not None else None)
        _emit(progress, 100, "done",
              f"Hoàn tất upscale ảnh {scale}x -> {tw}x{th} ({time.time() - started:.1f}s)")
        return output_path

    # ------------------------------------------------------------------ video
    def upscale_video(self, input_path, output_path, scale, model_name=None, frame_format="jpg",
                      progress=None, stop_check=None, temp_dir=None):
        frame_ext = "png" if str(frame_format).lower() == "png" else "jpg"
        info = probe.get_media_info(input_path)
        fps = info["fps"] or 24.0
        duration = info["duration"] or 0.0
        total_estimate = info["nb_frames"] or int(duration * fps)
        final_w, final_h = calculate_video_dimensions(info["width"], info["height"], scale)

        key = self.use_model(model_name)
        native = models.MODEL_REGISTRY[key]["native"]

        work = Path(temp_dir or Path.cwd() / "temp") / f"job_{int(time.time() * 1000)}"
        frames_dir = work / "frames"
        upscaled_dir = work / "upscaled"
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        try:
            _emit(progress, 3, "extracting",
                  f"Trích xuất frames ({fps} fps, ~{total_estimate} frames, {frame_ext.upper()}"
                  f"{' + NVDEC' if self._use_nvdec() else ''})...")
            self._extract_frames(input_path, frames_dir, fps, frame_ext, progress=progress)

            frame_list = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() == f".{frame_ext}")
            total_frames = len(frame_list)
            if total_frames == 0:
                raise RuntimeError("Không trích xuất được frame nào từ video.")

            gpu_list = ", ".join(self.devices) if self.devices != ["cpu"] else "cpu"
            _emit(progress, 12, "upscaling",
                  f"AI Upscale {total_frames} frames ({scale}x, {key} native {native}x, "
                  f"{len(self.devices)} thiết bị: {gpu_list}, batch {self.effective_batch})...")
            self._upscale_frames(frame_list, upscaled_dir, frame_ext, key, total_frames, progress, stop_check)

            encoder = fftools.pick_video_encoder(self.encoder)
            _emit(progress, 88, "assembling", f"Ghép frames + giữ nguyên audio ({encoder.upper()}/AAC)...")
            self._assemble_video(
                input_path, upscaled_dir, output_path, frame_ext, fps, final_w, final_h,
                info["width"], info["height"], native, info["has_audio"], duration, progress, stop_check,
            )
            _emit(progress, 100, "done", f"Hoàn tất upscale video {scale}x -> {final_w}x{final_h}")
            return output_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _upscale_frames(self, frame_list, upscaled_dir, frame_ext, model_key, total, progress, stop_check):
        """Pipeline 3 tầng: đọc frame || suy luận (1 luồng/GPU) || ghi frame.

        Nhờ vậy GPU không phải chờ giải mã/ghi JPEG, kể cả khi chỉ có 1 GPU.
        """
        workers = [self._worker(device, model_key) for device in self.devices]
        batch = max(1, self.batch)
        groups = [frame_list[i : i + batch] for i in range(0, len(frame_list), batch)]
        save_kwargs = {"quality": 100} if frame_ext == "jpg" else {}

        errors = []
        failed = threading.Event()
        state = {"done": 0}
        lock = threading.Lock()
        started = time.time()
        capacity = max(2, len(workers) * 2)
        q_decode = queue.Queue(maxsize=capacity)
        q_result = queue.Queue(maxsize=capacity)
        sentinel = object()

        def put(queue_obj, item):
            while not failed.is_set():
                try:
                    queue_obj.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def get(queue_obj):
            while not failed.is_set():
                try:
                    return queue_obj.get(timeout=0.2)
                except queue.Empty:
                    continue
            return None

        def decoder():
            try:
                for group in groups:
                    _check_stop(stop_check)
                    arrays = []
                    for path in group:
                        with Image.open(path) as frame:
                            arrays.append(np.asarray(frame.convert("RGB"), dtype=np.uint8))
                    if not put(q_decode, (group, arrays)):
                        return
            except BaseException as exc:  # noqa: BLE001 - chuyển lỗi sang luồng chính
                errors.append(exc)
                failed.set()
            finally:
                for _ in workers:
                    put(q_decode, sentinel)

        def infer(worker):
            try:
                while True:
                    item = get(q_decode)
                    if item is None or item is sentinel:
                        return
                    group, arrays = item
                    outs = self.upscale_arrays(arrays, model_name=model_key, worker=worker)
                    if not put(q_result, (worker, group, outs)):
                        return
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                failed.set()
            finally:
                put(q_result, sentinel)

        def writer():
            finished = 0
            while finished < len(workers):
                item = get(q_result)
                if item is None:
                    return
                if item is sentinel:
                    finished += 1
                    continue
                worker, group, outs = item
                for path, array in zip(group, outs):
                    _numpy_to_pil(array).save(upscaled_dir / path.name, **save_kwargs)
                with lock:
                    state["done"] += len(group)
                    count = state["done"]
                elapsed = max(1e-6, time.time() - started)
                speed = count / elapsed
                eta = int((total - count) / speed) if speed > 0 else 0
                _emit(progress, 12 + 73 * count / total, "upscaling",
                      f"AI Upscale: {count}/{total} frames ({100 * count / total:.1f}%) "
                      f"• {speed:.1f} fps • {worker.device} • còn ~{eta}s")

        threads = [threading.Thread(target=decoder, name="valo-decode")]
        threads += [threading.Thread(target=infer, args=(worker,), name=f"valo-infer-{worker.device}")
                    for worker in workers]
        threads += [threading.Thread(target=writer, name="valo-write")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

    def _use_nvdec(self):
        return self.hwaccel and any(device.startswith("cuda") for device in self.devices) \
            and fftools.supports_cuda_decode()

    def _extract_frames(self, input_path, frames_dir, fps, frame_ext, progress=None):
        ffmpeg = fftools.require_ffmpeg()
        quality = VIDEO_FRAME_QUALITY.get(frame_ext, [])
        pattern = str(frames_dir / f"frame_%06d.{frame_ext}")
        args = ["-i", str(input_path), "-vf", f"fps={fps}"] + quality + [pattern]

        attempts = []
        if self._use_nvdec():
            attempts.append([ffmpeg, "-y", "-hwaccel", "cuda"] + args)
        attempts.append([ffmpeg, "-y"] + args)

        last_error = "không rõ"
        for index, cmd in enumerate(attempts):
            code, _, err = fftools.run(cmd)
            if code == 0:
                return
            last_error = err[-300:]
            if index + 1 < len(attempts):
                _emit(progress, 4, "extracting", "NVDEC lỗi, chuyển sang giải mã CPU...")
        raise RuntimeError(f"Trích xuất frames thất bại: {last_error}")

    def _assemble_video(self, input_path, upscaled_dir, output_path, frame_ext, fps, final_w, final_h,
                        src_w, src_h, native, has_audio, duration, progress, stop_check):
        ffmpeg = fftools.require_ffmpeg()
        scale_vf = f"scale={final_w}:{final_h}:flags=lanczos,setsar=1"
        if final_w == src_w * native and final_h == src_h * native:
            scale_vf = "setsar=1"

        base = ["-y", "-framerate", str(fps), "-i", str(upscaled_dir / f"frame_%06d.{frame_ext}"),
                "-i", str(input_path), "-map", "0:v"]
        audio = (
            ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "aresample=async=1:first_pts=0"]
            if has_audio else ["-an"]
        )

        encoder = fftools.pick_video_encoder(self.encoder)
        passes = []
        if encoder == "h264_nvenc":  # GPU encode: nhanh hơn nhiều khi khung hình lớn (4K)
            passes.append(dict(
                pre=[],
                vf=scale_vf,
                enc=["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p",
                     "-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
                tail=["-progress", "pipe:1"],
            ))
        passes += [
            dict(
                pre=[],
                vf=scale_vf,
                enc=["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                     "-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
                tail=["-progress", "pipe:1"],
            ),
            dict(
                pre=["-fflags", "+genpts+discardcorrupt"],
                vf=f"scale={final_w}:{final_h}:flags=lanczos,setsar=1",
                enc=["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                     "-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
                tail=["-progress", "pipe:1"],
            ),
            dict(
                pre=["-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt",
                     "-avoid_negative_ts", "make_zero"],
                vf=f"scale={final_w}:{final_h}:flags=bicubic,setsar=1",
                enc=["-c:v", "libx264", "-crf", "22", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     "-max_muxing_queue_size", "4096"],
                tail=["-progress", "pipe:1"],
            ),
        ]

        last_error = None
        for index, spec in enumerate(passes, start=1):
            _check_stop(stop_check)
            cmd = [ffmpeg] + base + spec["pre"] + ["-vf", spec["vf"]] + audio + spec["enc"] + spec["tail"] \
                + [str(output_path)]
            code, _, err = fftools.run(
                cmd,
                on_stdout_line=lambda line: _ffmpeg_progress(line, duration, progress, 88, 99,
                                                             f"Ghép video (pass {index})"),
            )
            if code == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
                return
            last_error = f"pass {index}: mã {code} — {err[-200:] if err else ''}"
            if progress:
                _emit(progress, 88, "assembling", f"Pass {index} lỗi, thử chế độ cứu hộ pass {index + 1}...")
        raise RuntimeError(f"Ghép video thất bại sau {len(passes)} pass: {last_error}")


def _ffmpeg_progress(line, duration, progress, start_percent, end_percent, label):
    if not progress or not duration:
        return
    for key in ("out_time_us=", "out_time_ms="):
        if line.startswith(key):
            try:
                value = int(line.split("=", 1)[1].strip())
            except ValueError:
                return
            seconds = value / 1_000_000 if key == "out_time_us=" else value / 1_000
            ratio = min(1.0, seconds / duration)
            _emit(progress, start_percent + (end_percent - start_percent) * ratio, "assembling",
                  f"{label}: {ratio * 100:.1f}%")
            return


# ====================================================================== FFmpeg HQ


def _encoder_args(crf=18, preset="fast", encoder="auto"):
    if fftools.pick_video_encoder(encoder) == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", str(min(51, crf + 1)), "-b:v", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]


def upscale_image_ffmpeg(input_path, output_path, scale, progress=None, stop_check=None):
    """"Thuật toán FFmpeg Ultra HQ": Lanczos + unsharp (thuần CPU)."""
    ffmpeg = fftools.require_ffmpeg()
    info = probe.get_media_info(input_path)
    width, height = calculate_image_dimensions(info["width"] or 1024, info["height"] or 768, scale)
    _emit(progress, 25, "processing", f"FFmpeg Ultra HQ ({width}x{height} • {scale}x Lanczos + Unsharp)...")
    _check_stop(stop_check)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={width}:{height}:flags=lanczos,unsharp=3:3:0.5:3:3:0.2"
    code, _, err = fftools.run([ffmpeg, "-y", "-i", str(input_path), "-vf", vf, str(output_path)])
    if code != 0 or not Path(output_path).exists():
        _emit(progress, 60, "processing", "Unsharp lỗi, chuyển sang Lanczos thuần...")
        code, _, err = fftools.run(
            [ffmpeg, "-y", "-i", str(input_path), "-vf", f"scale={width}:{height}:flags=lanczos", str(output_path)]
        )
        if code != 0:
            raise RuntimeError(f"FFmpeg HQ thất bại (mã {code}): {err[-300:]}")
    _emit(progress, 100, "done", f"Hoàn tất upscale ảnh {scale}x ({width}x{height})")
    return output_path


def upscale_video_ffmpeg(input_path, output_path, scale, progress=None, stop_check=None, encoder="auto"):
    """FFmpeg trực tiếp trên video, 3 pass tự cứu giống app gốc (có thể dùng NVENC)."""
    ffmpeg = fftools.require_ffmpeg()
    info = probe.get_media_info(input_path)
    duration = info["duration"] or 1.0
    final_w, final_h = calculate_video_dimensions(info["width"], info["height"], scale)
    audio = (
        ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "aresample=async=1:first_pts=0"]
        if info["has_audio"]
        else ["-an"]
    )

    _emit(progress, 5, "processing", f"FFmpeg Ultra HQ ({final_w}x{final_h} • {scale}x Lanczos)...")
    passes = [
        dict(
            pre=[],
            vf=f"scale={final_w}:{final_h}:flags=lanczos,unsharp=3:3:0.4:3:3:0.2,setsar=1",
            enc=_encoder_args(18, "fast", encoder) + ["-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
            audio_extra=audio,
        ),
        dict(
            pre=["-fflags", "+genpts+discardcorrupt"],
            vf=f"scale={final_w}:{final_h}:flags=lanczos,setsar=1",
            enc=["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                 "-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
            audio_extra=["-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                         "-af", "aresample=async=1:first_pts=0"] if info["has_audio"] else ["-an"],
        ),
        dict(
            pre=["-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt", "-avoid_negative_ts", "make_zero"],
            vf=f"scale={final_w}:{final_h}:flags=bicubic,setsar=1",
            enc=["-c:v", "libx264", "-crf", "22", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                 "-max_muxing_queue_size", "4096"],
            audio_extra=["-c:a", "aac", "-b:a", "128k"] if info["has_audio"] else ["-an"],
        ),
    ]

    last_error = None
    for index, spec in enumerate(passes, start=1):
        _check_stop(stop_check)
        cmd = (
            [ffmpeg, "-y"]
            + spec["pre"]
            + ["-i", str(input_path), "-vf", spec["vf"]]
            + spec["audio_extra"]
            + spec["enc"]
            + ["-progress", "pipe:1", str(output_path)]
        )
        code, _, err = fftools.run(
            cmd, on_stdout_line=lambda line: _ffmpeg_progress(line, duration, progress, 5, 99, f"FFmpeg pass {index}")
        )
        if code == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
            _emit(progress, 100, "done", f"Hoàn tất upscale video {scale}x ({final_w}x{final_h})")
            return output_path
        last_error = f"pass {index}: mã {code} — {err[-200:] if err else ''}"
        if progress:
            _emit(progress, 20 * index, "processing", f"Pass {index} lỗi, thử pass {index + 1}...")
    raise RuntimeError(f"Upscale video thất bại sau 3 pass: {last_error}")


# ====================================================================== dispatcher


def output_filename(input_path, scale, is_video, out_dir):
    stem = Path(input_path).stem
    ext = ".mp4" if is_video else ".png"
    return Path(out_dir) / f"{stem}_upscaled_{scale}x_{int(time.time() * 1000)}{ext}"


_ENGINE_CACHE: dict[tuple, RealESRGANEngine] = {}
_ENGINE_CACHE_LOCK = threading.Lock()
_MAX_CACHED_ENGINES = 2


def get_engine(model=None, tile=0, tile_pad=10, fp32=False, devices=None, batch=DEFAULT_BATCH,
               encoder="auto", hwaccel=True):
    """Cache engine theo cấu hình — tránh nạp lại model cho từng file trong batch."""
    key = (models.resolve_model_name(model) if model else None, int(tile or 0), int(tile_pad or 10),
           bool(fp32), str(devices), int(batch or 1), str(encoder), bool(hwaccel))
    with _ENGINE_CACHE_LOCK:
        engine_obj = _ENGINE_CACHE.get(key)
        if engine_obj is None:
            while len(_ENGINE_CACHE) >= _MAX_CACHED_ENGINES:
                _, old = _ENGINE_CACHE.popitem()
                old.close()
            engine_obj = RealESRGANEngine(model=model, tile=tile, tile_pad=tile_pad, fp32=fp32,
                                          devices=devices, batch=batch, encoder=encoder, hwaccel=hwaccel)
            _ENGINE_CACHE[key] = engine_obj
        return engine_obj


def process_media(input_path, out_dir="outputs", scale=4, model=None, engine="realesrgan", tile=0, tile_pad=10,
                  fp32=False, frame_format="jpg", device=None, devices=None, batch=DEFAULT_BATCH,
                  encoder="auto", hwaccel=True, progress=None, stop_check=None, temp_dir=None):
    """Upscale 1 file (ảnh hoặc video). Trả dict: output_path, engine_used, info, elapsed."""
    input_path = str(Path(input_path).resolve())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_video = probe.media_kind(input_path) == "video"
    scale = int(scale)
    model = models.resolve_model_name(model or (models.DEFAULT_VIDEO_MODEL if is_video else models.DEFAULT_IMAGE_MODEL))
    output_path = output_filename(input_path, scale, is_video, out_dir)

    started = time.time()
    engine_used = engine
    if engine == "ffmpeg_hq":
        if is_video:
            upscale_video_ffmpeg(input_path, output_path, scale, progress, stop_check, encoder=encoder)
        else:
            upscale_image_ffmpeg(input_path, output_path, scale, progress, stop_check)
    else:
        chosen_devices = devices if devices is not None else (device if device else None)
        ai = get_engine(model=model, tile=tile, tile_pad=tile_pad, fp32=fp32, devices=chosen_devices,
                        batch=batch, encoder=encoder, hwaccel=hwaccel)
        try:
            if is_video:
                ai.upscale_video(input_path, output_path, scale, model, frame_format=frame_format,
                                 progress=progress, stop_check=stop_check, temp_dir=temp_dir)
            else:
                ai.upscale_image(input_path, output_path, scale, model, progress=progress, stop_check=stop_check)
        except UpscaleCancelled:
            raise
        except Exception as exc:  # tự động rơi về FFmpeg HQ như app gốc
            if progress:
                _emit(progress, 15, "fallback", f"⚠️ Real-ESRGAN lỗi ({exc}). Chuyển sang FFmpeg Ultra HQ...")
            engine_used = "ffmpeg_hq"
            if is_video:
                upscale_video_ffmpeg(input_path, output_path, scale, progress, stop_check, encoder=encoder)
            else:
                upscale_image_ffmpeg(input_path, output_path, scale, progress, stop_check)

    return {
        "input_path": input_path,
        "output_path": str(output_path),
        "engine_used": engine_used,
        "media_type": "video" if is_video else "image",
        "scale": scale,
        "model": model,
        "info": probe.get_media_info(output_path),
        "elapsed": time.time() - started,
    }
