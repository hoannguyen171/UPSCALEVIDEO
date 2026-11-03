"""Engine upscale — bản port Python của src/upscaler.js trong app gốc.

Hai engine giống app:

1. ``realesrgan`` (AI, GPU CUDA): suy luận Real-ESRGAN bằng PyTorch, có tile/overlap
   để không tràn VRAM; ảnh native 4x rồi resample Lanczos (+unsharp) cho các mốc
   2x/3x/6x/8x, giống cách app dùng ncnn native scale rồi ffmpeg tinh chỉnh.
2. ``ffmpeg_hq`` (CPU, không cần GPU): scale Lanczos + unsharp, video có chuỗi 3 pass
   tự cứu (HQ -> Safe -> Rescue) đúng như bản JS.

``process_media`` là dispatcher: engine AI lỗi -> tự động rơi về ffmpeg_hq.
"""

import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import fftools, models, probe
from .dims import calculate_image_dimensions, calculate_video_dimensions

IMAGE_UNSHARP = dict(radius=3, percent=30, threshold=3)
VIDEO_FRAME_QUALITY = {"jpg": ["-q:v", "1"], "png": ["-pix_fmt", "rgb24"]}


class UpscaleCancelled(RuntimeError):
    pass


def _emit(progress, percent, stage, text):
    if progress:
        progress(int(percent), stage, text)


def _check_stop(stop_check):
    if stop_check and stop_check():
        raise UpscaleCancelled("Đã hủy theo yêu cầu.")


def _pil_to_numpy(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _numpy_to_pil(arr):
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _target_size(orig_size, scale, is_video):
    w, h = orig_size
    return calculate_video_dimensions(w, h, scale) if is_video else calculate_image_dimensions(w, h, scale)


class RealESRGANEngine:
    """Suy luận Real-ESRGAN trên PyTorch (CUDA nếu có)."""

    def __init__(self, model=None, tile=0, tile_pad=10, fp32=False, device=None):
        self.model_name = models.resolve_model_name(model or models.DEFAULT_IMAGE_MODEL)
        self.tile = int(tile or 0)
        self.tile_pad = int(tile_pad or 10)
        self.fp32 = bool(fp32)
        self.device = device or self._auto_device()
        self._cache = {}

    @staticmethod
    def _auto_device():
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def half(self):
        if self.fp32 or self.device == "cpu":
            return False
        return self.device in ("cuda", "mps")

    def use_model(self, model_name):
        key = models.resolve_model_name(model_name)
        if key != self.model_name:
            self.model_name = key

    def net(self, model_name=None):
        key = models.resolve_model_name(model_name or self.model_name)
        if key not in self._cache:
            net, native = models.load_model(key, device=self.device, half=self.half)
            if native != getattr(net, "out_scale", native):
                raise RuntimeError(f"Hệ số native của {key} không nhất quán ({native} vs {net.out_scale}).")
            self._cache[key] = (net, native)
        return self._cache[key]

    # ------------------------------------------------------------------ suy luận
    def _forward(self, net, patch):
        import torch

        divisor = getattr(net, "divisor", 1)
        out_scale = getattr(net, "out_scale", 4)
        h, w = patch.shape[:2]
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h or pad_w:  # pixel_unshuffle (RRDBNet 2x/1x) yêu cầu cạnh chia hết cho divisor
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        with torch.no_grad():
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
            if self.half:
                tensor = tensor.half()
            out = net(tensor)
            out = torch.clamp(out.float(), 0.0, 1.0)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return out[: h * out_scale, : w * out_scale]

    def infer(self, img, model_name=None):
        """img: float32 HxWx3 [0,1] -> float32 (H*native)x(W*native)x3."""
        import torch

        net, native = self.net(model_name)
        h, w = img.shape[:2]

        tile = self.tile
        while True:
            try:
                if tile <= 0 or (h <= tile and w <= tile):
                    return self._forward(net, img)
                out = np.zeros((h * native, w * native, 3), dtype=np.float32)
                for y in range(0, h, tile):
                    for x in range(0, w, tile):
                        y1, x1 = min(y + tile, h), min(x + tile, w)
                        py0, px0 = max(0, y - self.tile_pad), max(0, x - self.tile_pad)
                        py1, px1 = min(h, y1 + self.tile_pad), min(w, x1 + self.tile_pad)
                        patch_out = self._forward(net, img[py0:py1, px0:px1])
                        oy, ox = (y - py0) * native, (x - px0) * native
                        out[y * native : y1 * native, x * native : x1 * native] = patch_out[
                            oy : oy + (y1 - y) * native, ox : ox + (x1 - x) * native
                        ]
                return out
            except RuntimeError as exc:  # hết VRAM -> tự tăng tile rồi thử lại
                if "out of memory" not in str(exc).lower():
                    raise
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                if tile <= 0:
                    tile = 400
                else:
                    tile = tile // 2
                    if tile < 32:
                        raise RuntimeError(
                            "Hết VRAM ngay cả khi tile rất nhỏ. Hãy thử model nhẹ hơn "
                            "(realesr-animevideov3) hoặc chạy engine ffmpeg_hq."
                        ) from exc

    # ------------------------------------------------------------------ ảnh
    def upscale_image(self, input_path, output_path, scale, model_name=None, progress=None, stop_check=None):
        src = Image.open(input_path)
        alpha = None
        if src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info):
            alpha = src.convert("RGBA").getchannel("A")
        width, height = src.size

        self.use_model(model_name)
        native = models.MODEL_REGISTRY[self.model_name]["native"]
        _emit(progress, 5, "upscaling", f"AI Upscale {scale}x ({self.model_name}, native {native}x, {self.device})...")

        started = time.time()
        out = self.infer(_pil_to_numpy(src), self.model_name)
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
        _emit(
            progress,
            100,
            "done",
            f"Hoàn tất upscale ảnh {scale}x -> {tw}x{th} ({time.time() - started:.1f}s)",
        )
        return output_path

    # ------------------------------------------------------------------ video
    def upscale_video(
        self,
        input_path,
        output_path,
        scale,
        model_name=None,
        frame_format="jpg",
        progress=None,
        stop_check=None,
        temp_dir=None,
    ):
        frame_ext = "png" if str(frame_format).lower() == "png" else "jpg"
        info = probe.get_media_info(input_path)
        fps = info["fps"] or 24.0
        duration = info["duration"] or 0.0
        total_estimate = info["nb_frames"] or int(duration * fps)
        final_w, final_h = calculate_video_dimensions(info["width"], info["height"], scale)

        self.use_model(model_name)
        native = models.MODEL_REGISTRY[self.model_name]["native"]

        work = Path(temp_dir or Path.cwd() / "temp") / f"job_{int(time.time() * 1000)}"
        frames_dir = work / "frames"
        upscaled_dir = work / "upscaled"
        frames_dir.mkdir(parents=True, exist_ok=True)
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        try:
            _emit(progress, 3, "extracting", f"Trích xuất frames ({fps} fps, ~{total_estimate} frames, {frame_ext.upper()})...")
            self._extract_frames(input_path, frames_dir, fps, frame_ext)

            frame_list = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() == f".{frame_ext}")
            total_frames = len(frame_list)
            if total_frames == 0:
                raise RuntimeError("Không trích xuất được frame nào từ video.")

            _emit(
                progress,
                12,
                "upscaling",
                f"AI Upscale {total_frames} frames ({scale}x, {self.model_name} native {native}x, {self.device})...",
            )
            started = time.time()
            for index, frame_path in enumerate(frame_list, start=1):
                _check_stop(stop_check)
                with Image.open(frame_path) as frame:
                    up = self.infer(_pil_to_numpy(frame), self.model_name)
                out_img = _numpy_to_pil(up)
                save_kwargs = {"quality": 100} if frame_ext == "jpg" else {}
                out_img.save(upscaled_dir / frame_path.name, **save_kwargs)

                if index == total_frames or index % 5 == 0:
                    elapsed = max(1e-6, time.time() - started)
                    speed = index / elapsed
                    eta = int((total_frames - index) / speed) if speed > 0 else 0
                    _emit(
                        progress,
                        12 + 73 * index / total_frames,
                        "upscaling",
                        f"AI Upscale: {index}/{total_frames} frames ({100 * index / total_frames:.1f}%) "
                        f"• {speed:.1f} fps • còn ~{eta}s",
                    )

            _emit(progress, 88, "assembling", "Ghép frames + giữ nguyên audio (H.264/AAC)...")
            self._assemble_video(
                input_path, upscaled_dir, output_path, frame_ext, fps, final_w, final_h,
                info["width"], info["height"], native, info["has_audio"], duration, progress, stop_check,
            )
            _emit(progress, 100, "done", f"Hoàn tất upscale video {scale}x -> {final_w}x{final_h}")
            return output_path
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _extract_frames(self, input_path, frames_dir, fps, frame_ext):
        ffmpeg = fftools.require_ffmpeg()
        cmd = [ffmpeg, "-y", "-i", str(input_path), "-vf", f"fps={fps}"]
        cmd += VIDEO_FRAME_QUALITY[frame_ext] if frame_ext in VIDEO_FRAME_QUALITY else []
        cmd += [str(frames_dir / f"frame_%06d.{frame_ext}")]
        code, _, err = fftools.run(cmd)
        if code != 0:
            raise RuntimeError(f"Trích xuất frames thất bại (mã {code}): {err[-300:]}")

    def _assemble_video(
        self, input_path, upscaled_dir, output_path, frame_ext, fps, final_w, final_h,
        src_w, src_h, native, has_audio, duration, progress, stop_check,
    ):
        ffmpeg = fftools.require_ffmpeg()
        scale_vf = f"scale={final_w}:{final_h}:flags=lanczos,setsar=1"
        if final_w == src_w * native and final_h == src_h * native:
            scale_vf = "setsar=1"

        base = [
            "-y",
            "-framerate", str(fps),
            "-i", str(upscaled_dir / f"frame_%06d.{frame_ext}"),
            "-i", str(input_path),
            "-map", "0:v",
        ]
        audio = (
            ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "aresample=async=1:first_pts=0"]
            if has_audio
            else ["-an"]
        )

        passes = [
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
                pre=["-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt", "-avoid_negative_ts", "make_zero"],
                vf=f"scale={final_w}:{final_h}:flags=bicubic,setsar=1",
                enc=["-c:v", "libx264", "-crf", "22", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                     "-max_muxing_queue_size", "4096"],
                tail=["-progress", "pipe:1"],
            ),
        ]

        last_error = None
        for index, spec in enumerate(passes, start=1):
            _check_stop(stop_check)
            cmd = (
                [ffmpeg]
                + base
                + spec["pre"]
                + ["-vf", spec["vf"]]
                + audio
                + spec["enc"]
                + spec["tail"]
                + [str(output_path)]
            )
            code, _, err = fftools.run(
                cmd,
                on_stdout_line=lambda line: _ffmpeg_progress(line, duration, progress, 88, 99, f"Ghép video (pass {index})"),
            )
            if code == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
                return
            last_error = f"pass {index}: mã {code} — {err[-200:] if err else ''}"
            if progress:
                _emit(progress, 88, "assembling", f"Pass {index} lỗi, thử chế độ cứu hộ pass {index + 1}...")
        raise RuntimeError(f"Ghép video thất bại sau 3 pass: {last_error}")


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
            _emit(progress, start_percent + (end_percent - start_percent) * ratio, "assembling", f"{label}: {ratio * 100:.1f}%")
            return


# ====================================================================== FFmpeg HQ


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


def upscale_video_ffmpeg(input_path, output_path, scale, progress=None, stop_check=None):
    """FFmpeg trực tiếp trên video, 3 pass tự cứu giống app gốc."""
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
            enc=["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                 "-max_muxing_queue_size", "4096", "-movflags", "+faststart"],
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


def process_media(
    input_path,
    out_dir="outputs",
    scale=4,
    model=None,
    engine="realesrgan",
    tile=0,
    tile_pad=10,
    fp32=False,
    frame_format="jpg",
    device=None,
    progress=None,
    stop_check=None,
    temp_dir=None,
):
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
            upscale_video_ffmpeg(input_path, output_path, scale, progress, stop_check)
        else:
            upscale_image_ffmpeg(input_path, output_path, scale, progress, stop_check)
    else:
        ai = RealESRGANEngine(model=model, tile=tile, tile_pad=tile_pad, fp32=fp32, device=device)
        try:
            if is_video:
                ai.upscale_video(
                    input_path, output_path, scale, model, frame_format=frame_format,
                    progress=progress, stop_check=stop_check, temp_dir=temp_dir,
                )
            else:
                ai.upscale_image(input_path, output_path, scale, model, progress=progress, stop_check=stop_check)
        except UpscaleCancelled:
            raise
        except Exception as exc:  # tự động rơi về FFmpeg HQ như app gốc
            if progress:
                _emit(progress, 15, "fallback", f"⚠️ Real-ESRGAN lỗi ({exc}). Chuyển sang FFmpeg Ultra HQ...")
            engine_used = "ffmpeg_hq"
            if is_video:
                upscale_video_ffmpeg(input_path, output_path, scale, progress, stop_check)
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
