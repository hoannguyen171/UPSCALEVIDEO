"""CLI: python -m valo_upscale.cli -i input.mp4 -s 4 -n realesr-animevideov3"""

import argparse
import glob
import sys
import time
from pathlib import Path

from . import __version__, engine, fftools, models, probe

MEDIA_EXT = set(probe.VIDEO_EXT) | set(probe.IMAGE_EXT)
SCALES = (2, 3, 4, 6, 8)


def expand_inputs(patterns):
    """Nhận file, thư mục (quét đệ quy) hoặc glob; trả list file media đã sắp xếp."""
    found = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found += [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in MEDIA_EXT]
        elif path.is_file():
            found.append(path)
        else:
            found += [Path(p) for p in sorted(glob.glob(pattern, recursive=True)) if Path(p).is_file()]
    unique, seen = [], set()
    for item in found:
        resolved = item.resolve()
        if resolved not in seen and resolved.suffix.lower() in MEDIA_EXT:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def make_progress_printer(quiet):
    state = {"percent": -100, "stage": None}

    def printer(percent, stage, text):
        if quiet:
            return
        if stage == "done":
            print(f"  [{percent:3d}%] {text}")
            state.update(percent=-100, stage=None)
            return
        if stage == state["stage"] and percent - state["percent"] < 5 and percent < 100:
            return
        state.update(percent=percent, stage=stage)
        print(f"  [{percent:3d}%] {text}")

    return printer


def build_parser():
    parser = argparse.ArgumentParser(
        prog="valo-upscale",
        description="Upscale ảnh/video bằng Real-ESRGAN (GPU) hoặc FFmpeg Lanczos (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python -m valo_upscale.cli anh.jpg -s 4\n"
            "  python -m valo_upscale.cli video.mp4 -s 2 -n realesr-animevideov3 --tile 400\n"
            "  python -m valo_upscale.cli ./thu_muc_anh -s 8 -o outputs --engine ffmpeg_hq\n"
            "  python -m valo_upscale.cli --list-models\n"
        ),
    )
    parser.add_argument("inputs", nargs="*", help="File / thư mục / glob đầu vào (ảnh: jpg png webp; video: mp4 mkv mov avi webm)")
    parser.add_argument("-i", "--input", action="append", default=[], dest="input_opts", help="Thêm đầu vào (dùng được nhiều lần)")
    parser.add_argument("-o", "--outdir", default="outputs", help="Thư mục kết quả (mặc định: outputs/)")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Hệ số phóng đại: 2, 3, 4, 6, 8 (mặc định 4)")
    parser.add_argument("-n", "--model", default=None, help="Model Real-ESRGAN (mặc định: x4plus cho ảnh, animevideov3 cho video)")
    parser.add_argument(
        "-e",
        "--engine",
        default="realesrgan",
        choices=["realesrgan", "ffmpeg_hq", "cpu"],
        help="realesrgan = AI GPU/CUDA; cpu = AI chạy CPU; ffmpeg_hq = Lanczos thuần CPU (mặc định realesrgan)",
    )
    parser.add_argument("-t", "--tile", type=int, default=0, help="Kích thước tile khi suy luận (0 = cả ảnh, tự giảm khi hết VRAM)")
    parser.add_argument("--tile-pad", type=int, default=10, help="Vùng chồng lấn giữa các tile (mặc định 10)")
    parser.add_argument("--fp32", action="store_true", help="Tắt half precision (dùng khi fp16 gây lỗi)")
    parser.add_argument("--frame-format", default="jpg", choices=["jpg", "png"], help="Định dạng frame trung gian của video (jpg nhanh hơn)")
    parser.add_argument("--devices", default="auto",
                        help="Thiết bị suy luận: auto (mọi GPU) | 0 | 0,1 | cpu (mặc định auto)")
    parser.add_argument("--batch", type=int, default=engine.DEFAULT_BATCH,
                        help=f"Số patch/frame mỗi lần forward — tăng để GPU đỡ đói (mặc định {engine.DEFAULT_BATCH}, tự hạ khi hết VRAM)")
    parser.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x264"],
                        help="Encoder video: auto (NVENC nếu có) | nvenc | x264 (mặc định auto)")
    parser.add_argument("--no-hwaccel", action="store_true", help="Tắt giải mã NVDEC (mặc định tự dò)")
    parser.add_argument("--temp-dir", default=None, help="Thư mục tạm (mặc định ./temp)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Chỉ in kết quả cuối")
    parser.add_argument("--list-models", action="store_true", help="Liệt kê model và trạng thái weight")
    parser.add_argument("--version", action="version", version=f"valo-upscale-gpu {__version__}")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_models:
        print(f"Thư mục weight: {models.weights_dir()}")
        models.list_models()
        return 0

    inputs = expand_inputs(list(args.inputs) + list(args.input_opts))
    if not inputs:
        print("Không tìm thấy file media nào. Xem trợ giúp: python -m valo_upscale.cli -h", file=sys.stderr)
        return 2

    if args.scale not in SCALES:
        print(f"Cảnh báo: scale {args.scale}x không nằm trong bộ mặc định {SCALES} — vẫn chạy bình thường.")

    if args.model:
        try:
            args.model = models.resolve_model_name(args.model)
        except ValueError as exc:
            print(f"Lỗi: {exc}", file=sys.stderr)
            return 2

    ffmpeg_path, ffprobe_path = fftools.refresh_paths()
    engine_name = "realesrgan" if args.engine == "cpu" else args.engine
    devices = args.devices
    if args.engine == "cpu":
        devices = "cpu"

    if not args.quiet:
        print(f"ffmpeg : {ffmpeg_path or 'KHÔNG TÌM THẤY'}")
        print(f"ffprobe: {ffprobe_path or 'KHÔNG TÌM THẤY'}")
        if engine_name == "realesrgan":
            import torch  # noqa: PLC0415

            chosen = engine.parse_devices(devices)
            names = []
            for device in chosen:
                if device.startswith("cuda"):
                    index = int(device.split(":")[1])
                    names.append(f"{device} ({torch.cuda.get_device_name(index)})")
                else:
                    names.append(device)
            print(f"device : {', '.join(names)} | batch {args.batch if chosen and chosen[0].startswith('cuda') else 1} | "
                  f"NVENC {'có' if fftools.supports_encoder('h264_nvenc') else 'không'} | "
                  f"NVDEC {'có' if fftools.supports_cuda_decode() else 'không'}")
        print(f"Số file: {len(inputs)} | scale {args.scale}x | engine {args.engine} | model {args.model or 'tự chọn'}\n")

    printer = make_progress_printer(args.quiet)

    ok, failed, started = [], [], time.time()
    for index, path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {path.name}")
        try:
            result = engine.process_media(
                str(path),
                out_dir=args.outdir,
                scale=args.scale,
                model=args.model,
                engine=engine_name,
                tile=args.tile,
                tile_pad=args.tile_pad,
                fp32=args.fp32,
                frame_format=args.frame_format,
                devices=devices,
                batch=args.batch,
                encoder=args.encoder,
                hwaccel=not args.no_hwaccel,
                progress=printer,
                temp_dir=args.temp_dir,
            )
            info = result["info"]
            print(
                f"  -> {result['output_path']}  ({info['width']}x{info['height']}, "
                f"engine {result['engine_used']}, {result['elapsed']:.1f}s)"
            )
            ok.append(result)
        except engine.UpscaleCancelled:
            print("  -> đã hủy.")
        except Exception as exc:
            print(f"  -> LỖI: {exc}", file=sys.stderr)
            failed.append((str(path), str(exc)))

    print(f"\nXong: {len(ok)} thành công, {len(failed)} lỗi, tổng {time.time() - started:.1f}s")
    for path, err in failed:
        print(f"  ✗ {path}: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
