"""Ghép mọi thứ thành 1 job: chia sẻ video → tunnel → đẩy kernel Kaggle → đợi → tải kết quả về.

Tách khỏi GUI để test được bằng CLI giả, và để dùng lại từ dòng lệnh (``winapp/cli.py``).
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import kaggle_client as kc
from .share import FileShareServer
from .tunnel import ensure_cloudflared, start_quick_tunnel, wait_public

def worker_template_path():
    """Đường dẫn ``kaggle_jobs/recavsr_worker.py`` — chạy từ mã nguồn hoặc từ file exe đã đóng gói."""
    candidates = []
    bundled = getattr(sys, "_MEIPASS", None)          # PyInstaller onefile
    if bundled:
        candidates.append(Path(bundled) / "kaggle_jobs" / "recavsr_worker.py")
    candidates.append(Path(__file__).resolve().parent.parent / "kaggle_jobs" / "recavsr_worker.py")
    for path in candidates:
        if path.is_file():
            return path
    return candidates[-1]


WORKER_TEMPLATE = worker_template_path()


@dataclass
class RunOptions:
    video: Path
    out_dir: Path = Path("ket_qua")
    scale: float = 4.0
    precision: str = "int8"
    color_fix: str = "none"
    kernels: str = "fast"
    encoder: str = "auto"
    seed: int = 42
    kv_offload: bool = False
    lowvram: bool = True
    output_name: str = ""
    kernel_slug: str = "valo-recavsr"
    kernel_private: bool = True
    work_dir: Path = None
    kaggle_exe: str = "kaggle"
    cloudflared: Path = None
    wait_interval: int = 20
    timeout: int = 6 * 3600
    share_token: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    public_check_timeout: int = 60   # giây đợi /health qua tunnel (0 = bỏ qua)
    drive_prefix: str = ""

    def worker_job(self, video_url):
        job = kc.JobSpec(video_url=video_url, scale=float(self.scale), precision=self.precision,
                         color_fix=self.color_fix, kernels=self.kernels, encoder=self.encoder, seed=int(self.seed),
                         kv_offload=bool(self.kv_offload), lowvram=bool(self.lowvram),
                         output_name=self.output_name or f"{Path(self.video).stem}_recavsr_x{self.scale:g}.mp4")
        return job.as_dict()


class Runner:
    """Chạy 1 job. Gọi ``prepare()`` rồi ``run()``; ``cleanup()`` luôn chạy ở cuối."""

    def __init__(self, options: RunOptions, log=None, on_state=None, kaggle=None, worker_template=None):
        self.options = options
        self.log = log or (lambda text: None)
        self.on_state = on_state or (lambda state, detail=None: None)
        self.kaggle = kaggle or kc.KaggleCli(executable=options.kaggle_exe, log=self.log)
        self.template_path = Path(worker_template or WORKER_TEMPLATE)
        self.share = None
        self.tunnel = None
        self.kernel_dir = None
        self.state = "idle"

    # ------------------------------------------------------------- chuẩn bị
    def _work_dir(self):
        base = Path(self.options.work_dir or (Path(self.options.video).parent / "valo_job"))
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _set(self, state, detail=None):
        self.state = state
        self.on_state(state, detail)

    def prepare_tunnel(self):
        """Mở HTTP server cho video + tunnel công khai, trả URL video công khai."""
        opts = self.options
        video = Path(opts.video)
        if not video.is_file():
            raise FileNotFoundError(f"không thấy video: {video}")
        work = self._work_dir()

        self._set("sharing")
        self.share = FileShareServer(video, token=opts.share_token)
        self.share.start()
        self.log(f"Đang chia sẻ {video.name} ({video.stat().st_size / 1048576:.1f} MB) ở {self.share.local_url}")

        exe = opts.cloudflared or ensure_cloudflared(work / "bin", log=self.log)
        self._set("tunnel")
        self.tunnel = start_quick_tunnel(self.share.port, exe, log=self.log, log_path=work / "cloudflared.log")
        if opts.public_check_timeout and not wait_public(self.tunnel.url, timeout=opts.public_check_timeout):
            self.log("Cảnh báo: /health qua tunnel chưa trả ok — vẫn thử tiếp")
        public = self.share.public_video_url(self.tunnel.url)
        self.log(f"URL video công khai: {public}")
        return public

    def prepare(self, video_url=None, allow_placeholder_user=False):
        """Tạo URL công khai (hoặc dùng sẵn) + thư mục kernel. Trả dict để in ra/in dry-run.

        ``allow_placeholder_user``: bản xem trước vẫn dựng được kế hoạch khi chưa có ``kaggle.json``
        (chỉ để xem luồng; chạy thật bắt buộc phải có credentials).
        """
        opts = self.options
        video_url = video_url or self.prepare_tunnel()
        job = opts.worker_job(video_url)

        self._set("kernel")
        template = self.template_path.read_text(encoding="utf-8")
        try:
            username = self.kaggle.username()
        except kc.KaggleError:
            if not allow_placeholder_user:
                raise
            username = "TEN-KAGGLE-CUA-BAN"
            self.log("Chưa có kaggle.json → bản xem trước dùng tên tạm; chạy thật cần credentials Kaggle.")
        kernel_ref = f"{username}/{opts.kernel_slug}"
        self.kernel_dir = kc.build_kernel_dir(self._work_dir() / "kernel", template, job, kernel_ref,
                                              metadata=kc.kernel_metadata(kernel_ref, is_private=opts.kernel_private))
        self.log(f"Thư mục kernel: {self.kernel_dir} (ref {kernel_ref})")
        return {"video_url": video_url, "job": job, "kernel_ref": kernel_ref, "kernel_dir": str(self.kernel_dir),
                "tunnel_url": self.tunnel.url if self.tunnel else None}

    def dry_run(self):
        """Chuẩn bị đầy đủ (kể cả tunnel) nhưng không đẩy kernel — để xem trước.

        Vẫn phải dọn share/tunnel sau khi xem trước, nếu không sẽ bỏ lại 1 tunnel đang mở.
        """
        try:
            plan = self.prepare(allow_placeholder_user=True)
            plan["dry_run"] = True
            return plan
        finally:
            self.cleanup()

    # ------------------------------------------------------------- chạy
    def run(self, video_url=None):
        opts = self.options
        plan = self.prepare(video_url=video_url)
        try:
            self._set("push")
            self.log("Đang đẩy kernel lên Kaggle (lần đầu sẽ cài ComfyUI + tải weight 6,3 GB nên khá lâu)...")
            ref = self.kaggle.push(self.kernel_dir)
            self.log(f"Kernel: https://www.kaggle.com/code/{ref}")

            self._set("waiting")
            state = self.kaggle.wait(ref, on_progress=lambda st, seconds: self.on_state("waiting", f"{st} ({seconds}s)"),
                                     interval=opts.wait_interval, timeout=opts.timeout)

            # chỉ tải file cần (json/log/mp4): output của job ComfyUI có thể tới ~1500 file vài GB
            self._set("downloading")
            out_dir = self._work_dir() / "kaggle_output"
            fetch = getattr(self.kaggle, "fetch_files", None) or self.kaggle.output
            fetch(ref, out_dir)
            found = kc.find_result(out_dir)

            if state != "complete":
                reason = (found["result"] or {}).get("error") if found["result"] else None
                if found["log"]:
                    self.log(f"log worker ({Path(found['log']).name}, 12 dòng cuối):")
                    for line in Path(found["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-12:]:
                        self.log("   | " + line[:200])
                raise kc.KaggleError(f"Kernel kết thúc với trạng thái {state}"
                                     + (f" — lý do: {reason}" if reason else "")
                                     + " (mở log trên Kaggle để xem đầy đủ)")
            if not found["video"]:
                raise kc.KaggleError(f"Không thấy video kết quả trong {out_dir} "
                                     f"(result: {json.dumps(found['result'], ensure_ascii=False)[:200]})")
            target = kc.copy_result(found["video"], opts.out_dir)
            self._set("done")
            self.log(f"Xong: {target}")
            return {"ok": True, "video": str(target), "kernel_ref": ref, "result": found["result"],
                    "log": str(found["log"]) if found["log"] else None, **plan}
        finally:
            self.cleanup()

    def cleanup(self):
        if self.share:
            self.share.stop()
            self.share = None
        if self.tunnel:
            self.tunnel.stop()
            self.tunnel = None
        self._set("idle")
