"""Giao diện Gradio chạy trong Colab/Kaggle (thay cho giao diện Electron của app gốc).

    python -m valo_upscale.app                 # mở UI local
    python -m valo_upscale.app --share         # tạo link công khai (Colab/Kaggle)

Phần xử lý nằm ở ``upscale_batch`` (module-level, không phụ thuộc Gradio) để có thể
gọi trực tiếp từ notebook hoặc kiểm thử.
"""

import argparse
from pathlib import Path

from . import __version__, engine, models, probe

LOG_LINES = 14


def _log_tail(log):
    return "\n".join(log[-LOG_LINES:])


def upscale_batch(
    paths,
    out_dir="outputs",
    scale=4,
    model=None,
    engine_name="realesrgan",
    tile=0,
    tile_pad=10,
    fp32=False,
    frame_format="jpg",
    devices="auto",
    batch=engine.DEFAULT_BATCH,
    encoder="auto",
    hwaccel=True,
    progress_cb=None,
    stop_check=None,
    temp_dir=None,
):
    """Generator: mỗi bước yield dict {log, images, video, outputs, done, total}."""
    paths = [str(p) for p in paths]
    log, images, videos, outputs = [], [], [], []

    def snapshot(done):
        return {
            "log": _log_tail(log),
            "images": list(images),
            "video": videos[-1] if videos else None,
            "outputs": list(outputs),
            "done": done,
            "total": len(paths),
        }

    if not paths:
        log.append("Chưa chọn file nào.")
        yield snapshot(0)
        return

    for index, path in enumerate(paths, start=1):
        name = Path(path).name
        is_video = probe.media_kind(path) == "video"
        chosen = model
        if is_video and chosen in ("realesrgan-x4plus", "realesrgan-x4plus-anime"):
            chosen = models.DEFAULT_VIDEO_MODEL
        log.append(f"[{index}/{len(paths)}] {name} ({'video' if is_video else 'ảnh'})")
        yield snapshot(index - 1)

        def progress_cb_inner(percent, stage, text, _index=index):
            if progress_cb:
                progress_cb(min(0.99, ((_index - 1) + percent / 100) / len(paths)), f"{name}: {text}")

        try:
            result = engine.process_media(
                path,
                out_dir=out_dir,
                scale=int(scale),
                model=chosen,
                engine="realesrgan" if engine_name == "realesrgan" else "ffmpeg_hq",
                tile=int(tile),
                tile_pad=int(tile_pad),
                fp32=bool(fp32),
                frame_format=frame_format,
                devices=devices or "auto",
                batch=int(batch or engine.DEFAULT_BATCH),
                encoder=encoder,
                hwaccel=bool(hwaccel),
                progress=progress_cb_inner,
                stop_check=stop_check,
                temp_dir=temp_dir,
            )
        except Exception as exc:
            log.append(f"  ✗ LỖI: {exc}")
            yield snapshot(index)
            continue

        info = result["info"]
        log.append(
            f"  ✔ {Path(result['output_path']).name} — {info['width']}x{info['height']} "
            f"• {result['elapsed']:.1f}s • engine {result['engine_used']}"
        )
        outputs.append(result["output_path"])
        (videos if is_video else images).append(result["output_path"])
        yield snapshot(index)

    log.append(f"Hoàn tất {len(outputs)}/{len(paths)} file.")
    yield snapshot(len(paths))


def build_demo(out_dir="outputs", temp_dir=None, **launch_kwargs):
    import gradio as gr  # noqa: PLC0415

    def handler(files, scale, model, engine_name, tile, tile_pad, fp32, frame_format, devices, batch, encoder,
                hwaccel, progress=gr.Progress()):
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in (files or [])]
        last = None
        for state in upscale_batch(
            paths,
            out_dir=out_dir,
            scale=scale,
            model=model,
            engine_name=engine_name,
            tile=tile,
            tile_pad=tile_pad,
            fp32=fp32,
            frame_format=frame_format,
            devices=devices,
            batch=batch,
            encoder=encoder,
            hwaccel=hwaccel,
            progress_cb=lambda r, desc: progress(r, desc=desc),
            temp_dir=temp_dir,
        ):
            last = state
            yield state["log"], state["images"], state["video"], state["outputs"]
        if last is None:
            yield "Chưa chọn file nào.", [], None, []

    with gr.Blocks(title=f"Valo Upscale GPU {__version__}") as demo:
        gr.Markdown(
            "# Valo Upscale GPU\n"
            "Upscale ảnh/video bằng **Real-ESRGAN** (GPU CUDA) hoặc **FFmpeg Lanczos + Unsharp** (CPU).\n"
            "Chọn nhiều file cùng lúc cũng được — chúng sẽ được xử lý lần lượt."
        )
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Ảnh / video đầu vào", file_count="multiple", type="filepath")
                scale = gr.Radio([2, 3, 4, 6, 8], value=4, label="Hệ số phóng đại (Scale)")
                model = gr.Dropdown(
                    choices=list(models.MODEL_REGISTRY),
                    value=models.DEFAULT_IMAGE_MODEL,
                    label="Model Real-ESRGAN",
                )
                engine_name = gr.Dropdown(
                    choices=["realesrgan", "ffmpeg_hq"],
                    value="realesrgan",
                    label="Engine (realesrgan = AI GPU, ffmpeg_hq = Lanczos CPU)",
                )
                with gr.Accordion("Tinh chỉnh tốc độ / chất lượng", open=False):
                    devices = gr.Textbox(value="auto", label="Thiết bị (devices)",
                                         info="auto = mọi GPU · 0 · 0,1 · cpu (batch chỉ áp dụng cho GPU)")
                    batch = gr.Slider(1, 16, value=engine.DEFAULT_BATCH, step=1, label="Batch (patch/frame mỗi forward)")
                    tile = gr.Slider(0, 1024, value=0, step=32, label="Tile (0 = tự động, giảm khi hết VRAM)")
                    tile_pad = gr.Slider(0, 64, value=10, step=2, label="Tile padding (chồng lấn)")
                    encoder = gr.Dropdown(["auto", "nvenc", "x264"], value="auto", label="Encoder video (auto = NVENC nếu có)")
                    hwaccel = gr.Checkbox(True, label="Giải mã NVDEC (GPU decode)")
                    fp32 = gr.Checkbox(False, label="fp32 (tắt half precision)")
                    frame_format = gr.Dropdown(["jpg", "png"], value="jpg", label="Định dạng frame video")
                run = gr.Button("Bắt đầu Upscale", variant="primary")
            with gr.Column(scale=1):
                status = gr.Textbox(label="Tiến trình", lines=12, max_lines=12, autoscroll=True)
                gallery = gr.Gallery(label="Kết quả ảnh", columns=2, height=320)
                video = gr.Video(label="Kết quả video")
                outputs = gr.File(label="Tải kết quả", file_count="multiple")

        run.click(
            handler,
            inputs=[files, scale, model, engine_name, tile, tile_pad, fp32, frame_format, devices, batch, encoder,
                    hwaccel],
            outputs=[status, gallery, video, outputs],
        )

    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(prog="valo_upscale.app", description="Giao diện Gradio cho valo-upscale-gpu")
    parser.add_argument("-o", "--outdir", default="outputs")
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--share", action="store_true", help="Tạo link công khai (cần cho Colab/Kaggle)")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    demo = build_demo(out_dir=args.outdir, temp_dir=args.temp_dir)
    demo.queue().launch(share=args.share, server_port=args.port, quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
