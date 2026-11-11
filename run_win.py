"""Mở ứng dụng: upscale video bằng GPU Kaggle (ReCaVSR int8).

    python run_win.py                 # mở giao diện
    python run_win.py --self-test     # in thông tin môi trường + đường dẫn worker rồi thoát
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from winapp.app import main  # noqa: E402


def self_test():
    import json
    import tkinter

    from winapp import job as jobmod
    from winapp.tunnel import platform_asset

    import shutil

    try:
        import kagglesdk  # noqa: F401

        sdk = True
    except ImportError:
        sdk = False
    report = {
        "python": sys.version.split()[0],
        "tkinter": tkinter.TkVersion,
        "frozen": bool(getattr(sys, "frozen", False)),
        "worker_template": str(jobmod.worker_template_path()),
        "worker_template_exists": jobmod.worker_template_path().is_file(),
        "cloudflared_asset": platform_asset()[0],
        "kagglesdk": sdk,                       # dùng để chỉ tải file log/kết quả thay vì cả output
        "kaggle_cli": shutil.which("kaggle"),   # cần cho push/status/output
        "cwd": str(Path.cwd()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if sys.stdout is None:      # exe --noconsole: không có stdout → ghi ra file cạnh exe
        target = Path(sys.executable).resolve().parent / "valo_self_test.json"
        target.write_text(text, encoding="utf-8")
        return 0 if report["worker_template_exists"] else 1
    print(text)
    return 0 if report["worker_template_exists"] else 1


def dump_worker(target=None):
    """Ghi bản worker đang được app dùng ra file — để đối chiếu với repo (kiểm exe có cập nhật không)."""
    import shutil

    from winapp import job as jobmod

    source = jobmod.worker_template_path()
    if target is None:
        text = source.read_text(encoding="utf-8")
        if sys.stdout is None:
            target = Path(sys.executable).resolve().parent / "worker_from_exe.py"
        else:
            print(text)
            return 0
    shutil.copy2(source, target)
    print(f"đã ghi worker ({source}) → {target}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--self-test" in args:
        raise SystemExit(self_test())
    if "--dump-worker" in args:
        index = args.index("--dump-worker")
        target = args[index + 1] if len(args) > index + 1 else None
        raise SystemExit(dump_worker(target))
    raise SystemExit(main())
