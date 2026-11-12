"""Giao diện Windows (Tkinter) — upscale video bằng GPU Kaggle, model ReCaVSR int8.

Chỉ cần Python chuẩn + CLI ``kaggle`` (``pip install kaggle`` + ``~/.kaggle/kaggle.json``).
Chạy: ``python run_win.py`` (hoặc ``python -m winapp.app``).
"""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, X, Y, BooleanVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from . import kaggle_client as kc
from .job import RunOptions, Runner

SCALES = ("4", "2", "1.5", "3")
PRECISIONS = ("int8", "bf16")
COLOR_FIXES = ("none", "adain", "wavelet")
KERNELS = ("fast", "exact")
DECODERS = ("auto", "flashdecoder", "vae")
ENCODERS = ("auto", "hevc nvenc", "h264 nvenc", "h264")
# Dạng ĐÚNG cho Tk trên Windows: mỗi nhãn đi với một LIST mẫu, không phải một chuỗi cách nhau dấu cách
VIDEO_FILETYPES = [("Video", ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm", "*.m4v", "*.mpg", "*.mpeg", "*.wmv")),
                   ("Tất cả", "*.*")]


class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("Valo Upscale — GPU Kaggle (ReCaVSR int8, Wan2.2 4x)")
        self.geometry("880x620")
        self.minsize(760, 520)
        self.queue = queue.Queue()
        self.thread = None
        self.last_video = None
        self.last_log = None
        self._build_ui()
        self.after(150, self._drain)

    # ------------------------------------------------------------- giao diện
    def _build_ui(self):
        pad = dict(padx=8, pady=4)

        files = ttk.LabelFrame(self, text="1. Video")
        files.pack(fill=X, **pad)
        self.video_var = StringVar()
        ttk.Entry(files, textvariable=self.video_var).pack(side=LEFT, fill=X, expand=True, padx=6, pady=6)
        ttk.Button(files, text="Chọn video…", command=self._pick_video).pack(side=RIGHT, padx=6)

        out = ttk.LabelFrame(self, text="2. Nơi lưu kết quả")
        out.pack(fill=X, **pad)
        self.out_var = StringVar(value=str(Path.cwd() / "ket_qua"))
        ttk.Entry(out, textvariable=self.out_var).pack(side=LEFT, fill=X, expand=True, padx=6, pady=6)
        ttk.Button(out, text="Chọn thư mục…", command=self._pick_outdir).pack(side=RIGHT, padx=6)

        opts = ttk.LabelFrame(self, text="3. Tham số (mặc định đã tốt cho T4 16 GB)")
        opts.pack(fill=X, **pad)
        grid = ttk.Frame(opts)
        grid.pack(fill=X, padx=6, pady=6)

        self.scale_var = StringVar(value="4")
        self.precision_var = StringVar(value="int8")
        self.color_var = StringVar(value="none")
        self.kernels_var = StringVar(value="fast")
        self.decoder_var = StringVar(value="auto")
        self.encoder_var = StringVar(value="auto")
        self.seed_var = StringVar(value="42")
        self.kv_var = BooleanVar(value=False)
        self.lowvram_var = BooleanVar(value=True)

        def combo(row, column, label, variable, values, width=14):
            ttk.Label(grid, text=label).grid(row=row, column=column, sticky="w", padx=4, pady=3)
            ttk.Combobox(grid, textvariable=variable, values=values, width=width, state="readonly") \
                .grid(row=row, column=column + 1, sticky="w", padx=4, pady=3)

        combo(0, 0, "Hệ số (scale)", self.scale_var, SCALES)
        combo(0, 2, "Model", self.precision_var, PRECISIONS)
        combo(0, 4, "Sửa màu", self.color_var, COLOR_FIXES)
        combo(1, 0, "Kernel", self.kernels_var, KERNELS)
        combo(1, 2, "Encoder", self.encoder_var, ENCODERS)
        combo(2, 4, "Decoder", self.decoder_var, DECODERS)
        ttk.Label(grid, text="Seed").grid(row=1, column=4, sticky="w", padx=4)
        ttk.Entry(grid, textvariable=self.seed_var, width=16).grid(row=1, column=5, sticky="w", padx=4)
        ttk.Checkbutton(grid, text="kv_offload (ít VRAM hơn)", variable=self.kv_var) \
            .grid(row=2, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(grid, text="lowvram (khuyên dùng trên T4)", variable=self.lowvram_var) \
            .grid(row=2, column=2, columnspan=2, sticky="w", padx=4)

        run = ttk.Frame(self)
        run.pack(fill=X, **pad)
        ttk.Button(run, text="Kiểm tra Kaggle", command=self._check_kaggle).pack(side=LEFT, padx=4)
        ttk.Button(run, text="Lấy kaggle.json", command=self._help_token).pack(side=LEFT, padx=4)
        ttk.Button(run, text="Xem trước (dry-run)", command=lambda: self._start(dry_run=True)).pack(side=LEFT, padx=4)
        self.start_button = ttk.Button(run, text="BẮT ĐẦU (GPU Kaggle)", command=lambda: self._start(dry_run=False))
        self.start_button.pack(side=LEFT, padx=4)
        ttk.Button(run, text="Kiểm tra ComfyUI ở máy", command=self._check_comfy).pack(side=LEFT, padx=4)
        self.local_button = ttk.Button(run, text="Chạy local", command=self._start_local)
        self.local_button.pack(side=LEFT, padx=4)

        status = ttk.Frame(self)
        status.pack(fill=X, **pad)
        self.status_var = StringVar(value="Sẵn sàng")
        ttk.Label(status, textvariable=self.status_var).pack(side=LEFT, padx=4)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill=X, padx=8)

        log_frame = ttk.LabelFrame(self, text="4. Nhật ký")
        log_frame.pack(fill=BOTH, expand=True, **pad)
        self.log_widget = ScrolledText(log_frame, height=14, wrap="word", state="disabled")
        self.log_widget.pack(fill=BOTH, expand=True, padx=4, pady=4)

        bottom = ttk.Frame(self)
        bottom.pack(fill=X, **pad)
        ttk.Button(bottom, text="Mở thư mục kết quả", command=self._open_dir).pack(side=LEFT, padx=4)
        ttk.Button(bottom, text="Mở video kết quả", command=self._open_video).pack(side=LEFT, padx=4)
        self.comfy_var = StringVar(value="http://127.0.0.1:8188")
        ttk.Label(bottom, text="ComfyUI local:").pack(side=LEFT, padx=(16, 2))
        ttk.Entry(bottom, textvariable=self.comfy_var, width=26).pack(side=LEFT)

    # ------------------------------------------------------------- tiện ích
    def log(self, text):
        self.queue.put(("log", str(text)))

    def _drain(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log_widget.configure(state="normal")
                    self.log_widget.insert(END, payload + "\n")
                    self.log_widget.see(END)
                    self.log_widget.configure(state="disabled")
                elif kind == "state":
                    state, detail = payload
                    self.status_var.set(f"{state}{(' — ' + detail) if detail else ''}")
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self._fail(payload)
        except queue.Empty:
            pass
        self.after(150, self._drain)

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        for widget in (self.start_button, self.local_button):
            widget.configure(state=state)
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _pick_video(self):
        """Mở hộp thoại chọn video.

        Hai lỗi thật đã gặp trên Windows: (1) ``filetypes`` phải là *danh sách* mẫu (``("*.mp4", "*.mkv"…)``),
        không phải một chuỗi cách nhau bằng dấu cách — chuỗi đó làm hộp thoại **không liệt kê file nào**;
        (2) thiếu ``parent`` thì hộp thoại có thể mở **sau** cửa sổ chính nên tưởng như bấm không ăn.
        """
        try:
            self.lift()
            self.focus_force()
            path = filedialog.askopenfilename(
                parent=self,
                title="Chọn video cần upscale",
                initialdir=str(Path(self.video_var.get().strip()).parent if self.video_var.get().strip() else Path.cwd()),
                filetypes=VIDEO_FILETYPES)
        except Exception as exc:                      # noqa: BLE001 - phải HIỆN lỗi ra, bản exe không có console
            self.log(f"LỖI mở hộp thoại chọn video: {type(exc).__name__}: {exc}")
            messagebox.showerror("Không mở được hộp thoại",
                                 f"{type(exc).__name__}: {exc}\n\nBạn vẫn có thể dán đường dẫn video vào ô bên trên.",
                                 parent=self)
            return
        if path:
            self.video_var.set(path)
            if not self.out_var.get().strip():
                self.out_var.set(str(Path(path).parent / "ket_qua"))

    def _pick_outdir(self):
        try:
            self.lift()
            self.focus_force()
            path = filedialog.askdirectory(parent=self, title="Chọn thư mục lưu kết quả",
                                           initialdir=self.out_var.get().strip() or str(Path.cwd()))
        except Exception as exc:                      # noqa: BLE001
            self.log(f"LỖI mở hộp thoại chọn thư mục: {type(exc).__name__}: {exc}")
            messagebox.showerror("Không mở được hộp thoại", f"{type(exc).__name__}: {exc}", parent=self)
            return
        if path:
            self.out_var.set(path)

    @staticmethod
    def _clean_path(text):
        """Chuẩn hoá đường dẫn người dùng gõ/dán.

        Windows *Copy as path* bọc trong dấu ngoặc kép, và người dùng hay dán kèm ``&`` của PowerShell —
        để nguyên thì ``Path(...).is_file()`` sai và app báo "Thiếu video" dù file có thật.
        """
        value = (text or "").strip().strip('"').strip("'").strip()
        value = value.lstrip("&").strip()
        return str(Path(value).expanduser()) if value else ""

    def _options(self):
        video = Path(self._clean_path(self.video_var.get()) or ".")
        if not video.is_file():
            messagebox.showerror("Thiếu video", "Hãy chọn file video trước (hoặc dán đường dẫn vào ô).")
            return None
        try:
            scale = float(self.scale_var.get())
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Tham số sai", "Scale và Seed phải là số.")
            return None
        return RunOptions(video=video, out_dir=Path(self._clean_path(self.out_var.get()) or "ket_qua"), scale=scale,
                          precision=self.precision_var.get(), color_fix=self.color_var.get(),
                          kernels=self.kernels_var.get(), decoder=self.decoder_var.get(),
                          encoder=self.encoder_var.get(), seed=seed,
                          kv_offload=self.kv_var.get(), lowvram=self.lowvram_var.get(),
                          work_dir=video.parent / "valo_job")

    # ------------------------------------------------------------- hành động
    def _check_kaggle(self):
        def work():
            cli = kc.KaggleCli(log=self.log)
            self.log("Đang kiểm tra CLI kaggle…")
            try:
                self.log("kaggle: " + cli.version())
                self.log("user  : " + cli.username())
                self.log("creds : " + str(cli.credentials_path()))
                self.log("OK — sẵn sàng chạy trên Kaggle")
            except kc.KaggleError as exc:
                self.log(f"LỖI: {exc}")
                self.log("Cài: pip install kaggle ; rồi tải kaggle.json vào %USERPROFILE%\\.kaggle\\kaggle.json")
        threading.Thread(target=work, daemon=True).start()

    def _help_token(self):
        """Mở trang Kaggle để tạo token và chỉ chỗ chép file cho đúng."""
        import webbrowser  # noqa: PLC0415

        target = kc.KaggleCli().credentials_path()
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        webbrowser.open("https://www.kaggle.com/settings")
        self.log("Đã mở https://www.kaggle.com/settings — mục API → 'Create New Token' (tải về kaggle.json)")
        self.log(f"Chép file đó vào: {target}")
        self.log("Cách khác: tạo file đó với nội dung {\"username\": \"tên-Kaggle\", \"key\": \"khoá-API\"}")
        if target.exists():
            self.log("✔ Đã thấy kaggle.json — bấm 'Kiểm tra Kaggle' để xác nhận.")

    def _check_comfy(self):
        def work():
            import sys  # noqa: PLC0415

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            try:
                from valo_upscale import recavsr  # noqa: PLC0415
            except ImportError:
                self.log("Bản .exe gọn không kèm tính năng kiểm ComfyUI ở máy (thiếu numpy/PIL).")
                self.log("Muốn dùng: chạy từ mã nguồn → python run_win.py (hoặc pip install numpy pillow).")
                return

            base = self.comfy_var.get().strip() or "http://127.0.0.1:8188"
            try:
                state = recavsr.check_ready(base, recavsr.transformer_name(self.precision_var.get()))
                self.log(f"ComfyUI {base}: {'OK' if state['ok'] else state}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"Không kiểm được ComfyUI ở {base}: {type(exc).__name__}: {exc}")
        threading.Thread(target=work, daemon=True).start()

    def _start(self, dry_run):
        options = self._options()
        if not options:
            return
        self._set_running(True)
        self.log("=" * 70)
        self.log(f"{'XEM TRƯỚC' if dry_run else 'BẮT ĐẦU'}: {options.video.name} • scale {options.scale:g} • "
                 f"{options.precision} • {options.kernels} • decoder {options.decoder}")

        def work():
            runner = Runner(options, log=self.log, on_state=lambda s, d=None: self.queue.put(("state", (s, d))))
            try:
                result = runner.dry_run() if dry_run else runner.run()
                self.queue.put(("done", result))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}"))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _start_local(self):
        """Chạy trên ComfyUI ở máy (nếu đã cài node ReCaVSR + weight) — để thử nhanh/không cần Kaggle."""
        options = self._options()
        if not options:
            return
        self._set_running(True)
        base = self.comfy_var.get().strip() or "http://127.0.0.1:8188"
        self.log(f"Chạy local qua ComfyUI {base} …")

        def work():
            import sys  # noqa: PLC0415

            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            try:
                from valo_upscale import recavsr  # noqa: PLC0415
            except ImportError:
                self.queue.put(("error", "Bản .exe gọn không kèm chế độ chạy local (thiếu numpy/PIL).\n"
                                         "Hãy dùng nút BẮT ĐẦU (GPU Kaggle), hoặc chạy từ mã nguồn:\n"
                                         "  python run_win.py"))
                return
            try:
                out = Path(options.out_dir) / (options.output_name or
                                               f"{options.video.stem}_recavsr_x{options.scale:g}.mp4")
                result = recavsr.upscale_video(
                    str(options.video), str(out), comfy_dir=str(Path(base).parent), base=base,
                    scale=options.scale, precision=options.precision, color_fix=options.color_fix,
                    kernels=options.kernels, decoder=options.decoder, encoder=options.encoder, seed=options.seed,
                    kv_offload=options.kv_offload,
                    progress=lambda percent, stage, text: self.log(f"[{percent:3d}%] {text}"))
                self.queue.put(("done", {"ok": True, "video": result["output_path"], "local": True}))
            except Exception as exc:  # noqa: BLE001
                self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _finish(self, result):
        self._set_running(False)
        if result.get("dry_run"):
            self.log("Kế hoạch sẵn sàng (dry-run): " + str(result.get("video_url", ""))[:120])
            self.status_var.set("Xem trước xong — bấm BẮT ĐẦU để chạy thật")
            return
        video = result.get("video")
        self.last_video = Path(video) if video else None
        self.last_log = result.get("log")
        self.status_var.set(f"Xong: {self.last_video.name if self.last_video else '?'}")
        if self.last_video:
            self.log(f"Video kết quả: {self.last_video}")
            if messagebox.askyesno("Hoàn tất", f"Đã upscale xong:\n{self.last_video}\n\nMở video ngay?"):
                self._open_video()

    def _fail(self, message):
        self._set_running(False)
        self.status_var.set("Lỗi — xem nhật ký")
        self.log("LỖI: " + message)
        messagebox.showerror("Lỗi", message.splitlines()[0][:400])

    def _open_dir(self):
        target = Path(self.out_var.get().strip() or ".")
        target.mkdir(parents=True, exist_ok=True)
        self._open(target)

    def _open_video(self):
        if self.last_video and self.last_video.exists():
            self._open(self.last_video)
        else:
            messagebox.showinfo("Chưa có kết quả", "Chưa có video kết quả nào.")

    @staticmethod
    def _open(path):
        import os  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass


def main():
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
