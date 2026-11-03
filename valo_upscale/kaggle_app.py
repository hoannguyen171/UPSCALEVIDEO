"""UI Kaggle: nhập link thư mục công khai -> tải về -> upscale -> đẩy kết quả ngược lên.

Chạy UI trong notebook (Kaggle cần bật Internet để `share=True` hoạt động):

    import valo_upscale.kaggle_app as ka
    ka.build_demo().queue().launch(share=True)

Chạy không giao diện (headless):

    python -m valo_upscale.kaggle_app --headless --url <link> --scale 4 \
        --upload-kind dropbox --upload-folder /Upscale/out --limit 3

Token đọc từ biến môi trường khi chạy headless:
    DROPBOX_TOKEN, GITHUB_TOKEN, GDRIVE_SA_JSON (nội dung JSON hoặc đường dẫn file).
"""

import argparse
import os
import sys
from pathlib import Path

from . import engine, models, probe, remote

LOG_LINES = 18
WORK_DIR = os.environ.get("VALO_WORK_DIR", "/kaggle/working" if Path("/kaggle/working").is_dir() else ".")
UPLOAD_KINDS = ["kaggle_local", "gdrive", "dropbox", "github"]
UPLOAD_LABELS = {
    "kaggle_local": "Lưu vào /kaggle/working (tải ở tab Output)",
    "gdrive": "Google Drive — cần Service Account JSON",
    "dropbox": "Dropbox — cần access token",
    "github": "GitHub — cần PAT (Contents: Read and write)",
}


def _log_tail(log):
    return "\n".join(log[-LOG_LINES:])


def run_job(
    source,
    *,
    scale=4,
    model=None,
    engine_name="realesrgan",
    tile=0,
    tile_pad=10,
    fp32=False,
    frame_format="jpg",
    limit=0,
    upload_kind="kaggle_local",
    upload_folder=None,
    dropbox_token=None,
    github_token=None,
    drive_sa_json=None,
    work_dir=WORK_DIR,
    out_dir=None,
    temp_dir=None,
    progress_cb=None,
    stop_check=None,
):
    """Generator: yield dict {log, images, video, outputs, uploaded, done, total}."""
    work = Path(work_dir)
    input_dir = work / "input"
    out_dir = Path(out_dir or work / "outputs")
    log, images, videos, outputs, uploaded = [], [], [], [], []
    total = 0

    def snapshot(done):
        return {
            "log": _log_tail(log),
            "images": list(images),
            "video": videos[-1] if videos else None,
            "outputs": list(outputs),
            "uploaded": list(uploaded),
            "done": done,
            "total": total,
        }

    try:
        provider = remote.detect_provider(source or "")
        log.append(f"🔗 Nguồn: {provider} — {source}")
        yield snapshot(0)

        log.append("Đang liệt kê file trong thư mục...")
        yield snapshot(0)
        items = remote.list_remote(
            source,
            dropbox_token=dropbox_token,
            github_token=github_token,
        )
        if not items:
            log.append("✗ Không tìm thấy file ảnh/video nào. Kiểm tra lại link (Drive phải để 'Anyone with the link').")
            yield snapshot(0)
            return

        if limit and limit > 0:
            items = items[:limit]
        total = len(items)
        log.append(f"Tìm thấy {total} file:")
        for item in items[:10]:
            log.append(f"  • {item.name} ({item.size_text})")
        if total > 10:
            log.append(f"  … và {total - 10} file khác")
        yield snapshot(0)

        log.append("Đang tải về...")
        yield snapshot(0)
        paths = remote.download_remote(
            items,
            input_dir,
            dropbox_token=dropbox_token,
            progress=log.append,
        )
        log.append(f"Đã tải {len(paths)} file vào {input_dir}")
        yield snapshot(0)

        for index, path in enumerate(paths, start=1):
            if stop_check and stop_check():
                log.append("⏹ Đã dừng theo yêu cầu.")
                yield snapshot(index - 1)
                return
            is_video = probe.media_kind(path) == "video"
            chosen = model
            if is_video and chosen in ("realesrgan-x4plus", "realesrgan-x4plus-anime"):
                chosen = models.DEFAULT_VIDEO_MODEL
            log.append(f"[{index}/{len(paths)}] {Path(path).name} ({'video' if is_video else 'ảnh'})")
            yield snapshot(index - 1)

            def on_progress(percent, stage, text, _index=index):
                if progress_cb:
                    progress_cb(min(0.95, ((_index - 1) + percent / 100) / len(paths)), f"{Path(path).name}: {text}")

            try:
                result = engine.process_media(
                    str(path),
                    out_dir=out_dir,
                    scale=int(scale),
                    model=chosen,
                    engine="realesrgan" if engine_name == "realesrgan" else "ffmpeg_hq",
                    tile=int(tile),
                    tile_pad=int(tile_pad),
                    fp32=bool(fp32),
                    frame_format=frame_format,
                    progress=on_progress,
                    stop_check=stop_check,
                    temp_dir=temp_dir,
                )
            except engine.UpscaleCancelled:
                log.append("⏹ Đã dừng theo yêu cầu.")
                yield snapshot(index - 1)
                return
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

        if outputs:
            log.append(f"Đang đẩy {len(outputs)} file kết quả lên: {UPLOAD_LABELS.get(upload_kind, upload_kind)}")
            yield snapshot(len(paths))
            links = remote.upload_results(
                outputs,
                kind=upload_kind,
                folder=upload_folder or str(work / "upscaled"),
                dropbox_token=dropbox_token,
                github_token=github_token,
                drive_sa_json=drive_sa_json,
                progress=log.append,
            )
            uploaded.extend(links)
            for link in links[:10]:
                log.append(f"  ↗ {link}")
            log.append(f"Hoàn tất: upscale {len(outputs)}/{len(paths)} file, đẩy lên {len(links)} file.")
        else:
            log.append("Không có file nào upscale thành công.")
    except Exception as exc:
        log.append(f"✗ {exc}")

    yield snapshot(total)


def list_only(source, dropbox_token=None, github_token=None):
    """Xem trước danh sách file (không tải)."""
    items = remote.list_remote(source, dropbox_token=dropbox_token, github_token=github_token)
    if not items:
        return "Không tìm thấy file ảnh/video nào trong link này."
    lines = [f"{len(items)} file:"]
    lines += [f"  • {item.name} ({item.size_text})" for item in items[:60]]
    if len(items) > 60:
        lines.append(f"  … và {len(items) - 60} file khác")
    return "\n".join(lines)


def build_demo(work_dir=WORK_DIR, temp_dir=None, **launch_kwargs):
    import gradio as gr  # noqa: PLC0415

    def preview(source, dropbox_token, github_token, progress=gr.Progress()):
        try:
            return list_only(source, dropbox_token, github_token)
        except Exception as exc:
            return f"✗ {exc}"

    def handler(source, scale, model, engine_name, tile, frame_format, limit, upload_kind,
                upload_folder, drive_sa_file, dropbox_token, github_token, drive_sa_json, progress=gr.Progress()):
        last = None
        for state in run_job(
            source,
            scale=scale,
            model=model,
            engine_name=engine_name,
            tile=tile,
            frame_format=frame_format,
            limit=int(limit or 0),
            upload_kind=upload_kind,
            upload_folder=upload_folder or None,
            dropbox_token=dropbox_token or None,
            github_token=github_token or None,
            drive_sa_json=drive_sa_file or drive_sa_json or None,
            work_dir=work_dir,
            temp_dir=temp_dir,
            progress_cb=lambda ratio, desc: progress(ratio, desc=desc),
        ):
            last = state
            yield state["log"], state["images"], state["video"], state["outputs"]
        if last is None:
            yield "Không có gì để chạy.", [], None, []

    with gr.Blocks(title="Valo Upscale — thư mục công khai") as demo:
        gr.Markdown(
            "# Valo Upscale — thư mục công khai\n"
            "Dán **link thư mục công khai** chứa video/ảnh → tải về → upscale bằng GPU → đẩy kết quả lên nơi bạn chọn.\n\n"
            "⚠️ **Link công khai chỉ đọc được.** Muốn đẩy kết quả *vào lại thư mục đó* thì phải có quyền ghi: "
            "Google Drive → Service Account JSON (share thư mục cho email đó, quyền Editor); "
            "Dropbox → access token; GitHub → PAT. Không có credential thì để chế độ "
            "`kaggle_local` và tải file ở tab **Output** sau khi Save version."
        )
        source = gr.Textbox(label="Link thư mục / danh sách URL", lines=2,
                            placeholder="https://drive.google.com/drive/folders/... | https://github.com/user/repo/tree/main/videos | https://.../index/")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    scale = gr.Radio([2, 3, 4, 6, 8], value=4, label="Scale")
                    limit = gr.Number(value=0, precision=0, label="Giới hạn số file (0 = tất cả)")
                model = gr.Dropdown(list(models.MODEL_REGISTRY), value=models.DEFAULT_IMAGE_MODEL, label="Model")
                engine_name = gr.Dropdown(["realesrgan", "ffmpeg_hq"], value="realesrgan",
                                          label="Engine (realesrgan = GPU, ffmpeg_hq = Lanczos CPU)")
                with gr.Accordion("Tinh chỉnh", open=False):
                    tile = gr.Slider(0, 1024, value=0, step=32, label="Tile (0 = tự động, giảm khi hết VRAM)")
                    frame_format = gr.Dropdown(["jpg", "png"], value="jpg", label="Frame video")
            with gr.Column():
                upload_kind = gr.Dropdown(UPLOAD_KINDS, value="kaggle_local",
                                          label="Đẩy kết quả lên", info="Đích nhận file sau khi upscale")
                upload_folder = gr.Textbox(
                    label="Thư mục đích",
                    placeholder="kaggle_local: /kaggle/working/upscaled · gdrive: link/ID thư mục · "
                                "dropbox: /Upscale/kedua · github: owner/repo/thu-muc",
                )
                with gr.Accordion("Credential (chỉ cần khi đẩy lên Drive/Dropbox/GitHub)", open=False):
                    drive_sa_file = gr.File(label="Google Drive — file Service Account JSON (.json)",
                                            file_count="single", type="filepath")
                    drive_sa_json = gr.Textbox(label="… hoặc dán JSON / đường dẫn tới file .json", type="password")
                    dropbox_token = gr.Textbox(label="Dropbox access token", type="password")
                    github_token = gr.Textbox(label="GitHub PAT (Contents: Read and write)", type="password")

        with gr.Row():
            preview_btn = gr.Button("Xem trước danh sách file")
            run_btn = gr.Button("Tải → Upscale → Đẩy lên", variant="primary")
        preview_box = gr.Textbox(label="Nội dung thư mục", lines=8, max_lines=12)

        with gr.Row():
            status = gr.Textbox(label="Tiến trình", lines=16, max_lines=16, autoscroll=True)
            with gr.Column():
                gallery = gr.Gallery(label="Kết quả ảnh", columns=2, height=280)
                video = gr.Video(label="Kết quả video")
        outputs = gr.File(label="File kết quả (tải về)", file_count="multiple")

        preview_btn.click(preview, [source, dropbox_token, github_token], [preview_box])
        run_btn.click(
            handler,
            inputs=[source, scale, model, engine_name, tile, frame_format, limit, upload_kind, upload_folder,
                    drive_sa_file, dropbox_token, github_token, drive_sa_json],
            outputs=[status, gallery, video, outputs],
        )
    return demo


def main(argv=None):
    parser = argparse.ArgumentParser(prog="valo_upscale.kaggle_app",
                                     description="Tải thư mục công khai -> upscale -> đẩy kết quả ngược lên")
    parser.add_argument("--headless", action="store_true", help="Chạy không giao diện (in log ra stdout)")
    parser.add_argument("--url", help="Link thư mục/file công khai")
    parser.add_argument("--list", action="store_true", help="Chỉ liệt kê file rồi thoát")
    parser.add_argument("-s", "--scale", type=int, default=4)
    parser.add_argument("-n", "--model", default=None)
    parser.add_argument("-e", "--engine", default="realesrgan", choices=["realesrgan", "ffmpeg_hq", "cpu"])
    parser.add_argument("-t", "--tile", type=int, default=0)
    parser.add_argument("--frame-format", default="jpg", choices=["jpg", "png"])
    parser.add_argument("--limit", type=int, default=0, help="Chỉ xử lý N file đầu (0 = tất cả)")
    parser.add_argument("--upload-kind", default="kaggle_local", choices=UPLOAD_KINDS)
    parser.add_argument("--upload-folder", default=None)
    parser.add_argument("--work-dir", default=WORK_DIR)
    parser.add_argument("--share", action="store_true", help="Mở UI Gradio với link công khai")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    tokens = dict(
        dropbox_token=os.environ.get("DROPBOX_TOKEN"),
        github_token=os.environ.get("GITHUB_TOKEN"),
        drive_sa_json=os.environ.get("GDRIVE_SA_JSON"),
    )

    if args.list:
        if not args.url:
            print("Cần --url", file=sys.stderr)
            return 2
        print(list_only(args.url, tokens["dropbox_token"], tokens["github_token"]))
        return 0

    if args.headless:
        if not args.url:
            print("Cần --url", file=sys.stderr)
            return 2
        for state in run_job(
            args.url,
            scale=args.scale,
            model=args.model,
            engine_name="realesrgan" if args.engine == "cpu" else args.engine,
            tile=args.tile,
            frame_format=args.frame_format,
            limit=args.limit,
            upload_kind=args.upload_kind,
            upload_folder=args.upload_folder,
            work_dir=args.work_dir,
            progress_cb=lambda percent, text: print(f"  [{percent * 100:3.0f}%] {text}",
                                                    flush=True),
            **tokens,
        ):
            print(state["log"].splitlines()[-1], flush=True)
        return 0

    build_demo(work_dir=args.work_dir).queue().launch(share=args.share, server_port=args.port, quiet=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
