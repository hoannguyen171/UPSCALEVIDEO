"""Chạy job không cần giao diện (dùng để gỡ lỗi, tự động hoá, hoặc test).

    python -m winapp.cli --video clip.mp4 --scale 4 --precision int8 --dry-run
    python -m winapp.cli --video clip.mp4 --out ket_qua --scale 2 --color-fix adain
    python -m winapp.cli --check                  # kiểm kaggle CLI + credentials + cloudflared
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import kaggle_client as kc
from .job import RunOptions, Runner
from .tunnel import ensure_cloudflared, platform_asset


def build_options(args):
    return RunOptions(
        video=Path(args.video), out_dir=Path(args.out), scale=args.scale, precision=args.precision,
        color_fix=args.color_fix, kernels=args.kernels, encoder=args.encoder, seed=args.seed,
        kv_offload=args.kv_offload, lowvram=not args.no_lowvram, output_name=args.output_name or "",
        kernel_slug=args.kernel_slug, work_dir=Path(args.work_dir) if args.work_dir else None,
        kaggle_exe=args.kaggle_exe, wait_interval=args.interval, timeout=args.timeout,
        cloudflared=Path(args.cloudflared) if args.cloudflared else None,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="winapp.cli", description="Upscale video bằng GPU Kaggle (ReCaVSR int8)")
    parser.add_argument("--video", help="Video cần upscale")
    parser.add_argument("--out", default="ket_qua", help="Thư mục lưu kết quả")
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument("--precision", default="int8", choices=["int8", "bf16"])
    parser.add_argument("--color-fix", default="none", choices=["none", "adain", "wavelet"])
    parser.add_argument("--kernels", default="fast", choices=["fast", "exact"])
    parser.add_argument("--encoder", default="auto", choices=["auto", "hevc nvenc", "h264 nvenc", "h264"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-offload", action="store_true")
    parser.add_argument("--no-lowvram", action="store_true")
    parser.add_argument("--output-name", default="")
    parser.add_argument("--kernel-slug", default="valo-recavsr")
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--kaggle-exe", default="kaggle", help="đường dẫn CLI kaggle (test dùng CLI giả)")
    parser.add_argument("--cloudflared", default=None, help="đường dẫn cloudflared đã có sẵn")
    parser.add_argument("--interval", type=int, default=20, help="giây giữa 2 lần hỏi trạng thái")
    parser.add_argument("--timeout", type=int, default=6 * 3600)
    parser.add_argument("--dry-run", action="store_true", help="Chuẩn bị mọi thứ nhưng không đẩy kernel")
    parser.add_argument("--check", action="store_true", help="Kiểm kaggle CLI + credentials + cloudflared rồi thoát")
    parser.add_argument("--tunnel-test", action="store_true",
                        help="Tự kiểm tunnel thật: tải cloudflared, mở quick tunnel cho 1 file và thử /health qua URL công khai")
    parser.add_argument("--json", action="store_true", help="In kết quả dạng JSON (cho script khác đọc)")
    args = parser.parse_args(argv)

    log = (lambda text: print(text, flush=True)) if not args.json else (lambda text: None)

    if args.check:
        cli = kc.KaggleCli(executable=args.kaggle_exe, log=log)
        asset, local_name = platform_asset()
        report = {"kaggle_version": None, "username": None, "credentials": cli.credentials_path().exists(),
                  "cloudflared_asset": asset, "python": sys.version.split()[0]}
        try:
            report["kaggle_version"] = cli.version()
        except kc.KaggleError as exc:
            report["kaggle_error"] = str(exc)
        try:
            report["username"] = cli.username()
        except kc.KaggleError as exc:
            report["username_error"] = str(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report.get("username") else 1

    if args.tunnel_test:
        from pathlib import Path as _Path  # noqa: PLC0415

        from .share import FileShareServer  # noqa: PLC0415
        from .tunnel import ensure_cloudflared, start_quick_tunnel, wait_public  # noqa: PLC0415

        sample = _Path(args.video) if args.video else _Path(__file__).resolve().parent.parent / "kaggle_jobs" / "recavsr_worker.py"
        work = _Path(args.work_dir or (_Path.cwd() / "valo_job"))
        exe = ensure_cloudflared(work / "bin", log=log)
        share = FileShareServer(sample, token="tunneltest")
        share.start()
        tunnel = start_quick_tunnel(share.port, exe, log=log, log_path=work / "cloudflared.log")
        try:
            ok = wait_public(tunnel.url, timeout=90)
            report = {"tunnel_url": tunnel.url, "share_port": share.port, "file": str(sample),
                      "health_ok": ok, "video_url": share.public_video_url(tunnel.url)}
            if ok:   # tải thử qua URL công khai, đúng cách worker trên Kaggle sẽ lấy video
                import urllib.request  # noqa: PLC0415

                with urllib.request.urlopen(report["video_url"], timeout=300) as response:
                    downloaded = len(response.read())
                report["downloaded_bytes"] = downloaded
                report["payload_ok"] = downloaded == sample.stat().st_size
        finally:
            tunnel.stop()
            share.stop()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["health_ok"] else 1

    if not args.video:
        parser.error("cần --video (hoặc --check/--tunnel-test)")

    options = build_options(args)
    runner = Runner(options, log=log, on_state=lambda state, detail=None: log(f"[{state}] {detail or ''}"))
    if args.json:
        runner.log = lambda text: None
    try:
        result = runner.dry_run() if args.dry_run else runner.run()
    except Exception as exc:  # noqa: BLE001 - in ra cho người dùng/script
        if args.json:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        else:
            print(f"LỖI: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, default=str))
    else:
        print("\nKẾ HOẠCH/JOB:" if args.dry_run else "\nKẾT QUẢ:")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
