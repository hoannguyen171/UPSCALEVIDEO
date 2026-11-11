"""Gọi Kaggle API bằng CLI ``kaggle``: đẩy kernel (worker), theo dõi trạng thái, tải kết quả về.

Không dùng SDK riêng để app chỉ cần ``pip install kaggle`` + ``~/.kaggle/kaggle.json`` (hoặc biến môi
trường ``KAGGLE_USERNAME``/``KAGGLE_KEY``). Mọi lệnh đều đi qua ``executable`` để test được bằng CLI giả.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

KAGGLE_STATUS = {
    "KernelWorkerStatus.COMPLETE": "complete",
    "KernelWorkerStatus.RUNNING": "running",
    "KernelWorkerStatus.QUEUED": "queued",
    "KernelWorkerStatus.ERROR": "error",
    "KernelWorkerStatus.CANCEL_REQUESTED": "cancelling",
    "KernelWorkerStatus.CANCEL_ACKNOWLEDGED": "cancelled",
}


class KaggleError(RuntimeError):
    pass


@dataclass
class JobSpec:
    """Tham số 1 job upscale (được nhét thẳng vào worker chạy trên Kaggle)."""

    video_url: str = ""
    scale: float = 4.0
    precision: str = "int8"
    color_fix: str = "none"
    kernels: str = "fast"
    encoder: str = "auto"
    seed: int = 42
    kv_offload: bool = False
    lowvram: bool = True
    output_name: str = ""
    worker_repo: str = "https://github.com/hoannguyen171/UPSCALEVIDEO.git"
    keep_comfy: bool = True
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        data = asdict(self)
        data.pop("extra", None)
        data.update(self.extra)
        return data


def render_worker(template_text, job):
    """Thay dòng ``JOB = {}`` trong worker bằng tham số thật (giữ nguyên phần còn lại)."""
    payload = json.dumps(job, ensure_ascii=False)
    lines = []
    replaced = False
    for line in template_text.splitlines():
        if line.strip() == "JOB = {}" and not replaced:
            lines.append(f"JOB = {payload}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise KaggleError("worker template thiếu dòng `JOB = {}` để chèn tham số")
    return "\n".join(lines) + "\n"


def kernel_metadata(kernel_ref, title="Valo ReCaVSR worker", code_file="recavsr_worker.py",
                    dataset_sources=None, enable_gpu=True, enable_internet=True, is_private=True):
    return {
        "id": kernel_ref,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": bool(is_private),
        "enable_gpu": bool(enable_gpu),
        "enable_internet": bool(enable_internet),
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_kernel_dir(dest, worker_text, job, kernel_ref, metadata=None):
    """Ghi thư mục kernel để ``kaggle kernels push`` dùng."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    code_file = "recavsr_worker.py"
    (dest / code_file).write_text(render_worker(worker_text, job), encoding="utf-8")
    meta = metadata or kernel_metadata(kernel_ref, code_file=code_file)
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


class KaggleCli:
    """Bọc CLI kaggle. ``executable`` có thể là script giả khi test."""

    def __init__(self, executable="kaggle", log=None, env=None, cwd=None):
        self.executable = executable
        self.log = log or (lambda text: None)
        self.env = dict(env or os.environ)
        self.cwd = cwd

    # ------------------------------------------------------------- chạy lệnh
    def _run(self, args, timeout=600, log_output=True):
        cmd = [str(self.executable)] + [str(a) for a in args]
        if log_output:
            self.log("$ " + " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout,
                                    env=self.env, cwd=str(self.cwd) if self.cwd else None)
        except FileNotFoundError as exc:
            raise KaggleError(f"Không chạy được `{self.executable}` — cài bằng `pip install kaggle` rồi đăng nhập "
                              f"(kaggle.json). Chi tiết: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise KaggleError(f"Lệnh kaggle quá thời gian {timeout}s: {' '.join(cmd)}") from exc
        output = (result.stdout or "") + (result.stderr or "")
        if log_output:
            for line in output.strip().splitlines()[-4:]:
                self.log("  | " + line[:200])
        return result.returncode, output

    # ------------------------------------------------------------- thông tin
    def version(self):
        code, output = self._run(["--version"], timeout=120)
        return output.strip().splitlines()[0] if output.strip() else ""

    def credentials_path(self):
        env_path = self.env.get("KAGGLE_CONFIG_DIR")
        base = Path(env_path) if env_path else Path.home() / ".kaggle"
        return base / "kaggle.json"

    def username(self):
        """Lấy username từ kaggle.json hoặc biến môi trường."""
        if self.env.get("KAGGLE_USERNAME"):
            return self.env["KAGGLE_USERNAME"]
        data = read_json(self.credentials_path()) or {}
        username = data.get("username")
        if not username:
            # không in log của lệnh này: khi chưa có kaggle.json thì CLI in cả traceback, gây rối mắt
            code, output = self._run(["config", "view"], timeout=120, log_output=False)
            match = re.search(r'"username"\s*:\s*"([^"]+)"', output)
            username = match.group(1) if match else None
        if not username:
            raise KaggleError("Chưa có kaggle.json — mở https://www.kaggle.com/settings (mục API → Create New Token), "
                              f"tải file rồi chép vào: {self.credentials_path()}  — hoặc đặt KAGGLE_USERNAME/KAGGLE_KEY")
        return username

    def has_credentials(self):
        try:
            self.username()
            return True
        except KaggleError:
            return False

    # ------------------------------------------------------------- kernel
    def push(self, kernel_dir, timeout=1800):
        code, output = self._run(["kernels", "push", "-p", str(kernel_dir)], timeout=timeout)
        if code != 0:
            raise KaggleError(f"kaggle kernels push thất bại (mã {code}): {output.strip()[-400:]}")
        match = re.search(r"https://www\.kaggle\.com/code/([\w.-]+/[\w.-]+)", output)
        if match:
            return match.group(1)
        meta = read_json(Path(kernel_dir) / "kernel-metadata.json") or {}
        if meta.get("id"):
            return meta["id"]
        raise KaggleError(f"Không đọc được kernel ref từ output: {output[-300:]}")

    def status(self, kernel_ref, timeout=300):
        code, output = self._run(["kernels", "status", kernel_ref], timeout=timeout)
        if code != 0:
            raise KaggleError(f"kaggle kernels status thất bại (mã {code}): {output.strip()[-300:]}")
        found = re.findall(r"KernelWorkerStatus\.\w+", output)
        if not found:
            return "unknown"
        return KAGGLE_STATUS.get(found[-1], found[-1].split(".")[-1].lower())

    def wait(self, kernel_ref, on_progress=None, interval=20, timeout=6 * 3600):
        started = time.time()
        last = None
        while time.time() - started < timeout:
            state = self.status(kernel_ref)
            if on_progress:
                on_progress(state, int(time.time() - started))
            if state != last:
                self.log(f"trạng thái kernel: {state}")
                last = state
            if state in ("complete", "error", "cancelled"):
                return state
            time.sleep(interval)
        raise KaggleError(f"Kernel chạy quá {timeout}s chưa xong")

    def output(self, kernel_ref, out_dir, timeout=3600):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        code, output = self._run(["kernels", "output", kernel_ref, "-p", str(out_dir)], timeout=timeout)
        if code != 0:
            raise KaggleError(f"kaggle kernels output thất bại (mã {code}): {output.strip()[-400:]}")
        return out_dir

    def log(self, kernel_ref, out_dir, timeout=600):
        """Tải log của kernel (kaggle kernels output cũng kèm file .log nếu worker ghi vào /kaggle/working)."""
        return self.output(kernel_ref, out_dir, timeout=timeout)


def find_result(out_dir):
    """Đọc ``valo_result.json`` (worker ghi) + file video trong thư mục output."""
    out_dir = Path(out_dir)
    result = read_json(out_dir / "valo_result.json")
    videos = sorted([p for p in out_dir.rglob("*.mp4") if p.is_file()],
                    key=lambda p: p.stat().st_size, reverse=True)
    return {"result": result, "videos": videos, "video": videos[0] if videos else None,
            "log": (out_dir / "valo_run.log") if (out_dir / "valo_run.log").exists() else None}


def copy_result(video, target_dir, name=None):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (name or Path(video).name)
    if Path(video).resolve() != target.resolve():
        shutil.copy2(video, target)
    return target
