# valo-upscale-gpu

Bản port **Python** của app desktop *Valo Upscale* (Electron, thư viện `app.asar` đã được
dịch ngược bằng REA) — chạy được trên **GPU của Google Colab / Kaggle**, hoặc CPU local.

Giữ nguyên hành vi upscale của bản gốc: Real-ESRGAN cho ảnh/video, clamp 4K khi xuất video,
clamp 8K khi xuất ảnh, giữ nguyên audio, và chuỗi fallback tự cứu khi lỗi.
Bỏ hoàn toàn phần khoá bản quyền (Google Apps Script, Machine ID, HMAC) vì đây là bản tự viết lại.

| Chức năng của app gốc | Trong repo này |
|---|---|
| Scale 2x / 3x / 4x / 6x / 8x | `--scale 2|3|4|6|8` |
| Model `realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-animevideov3` | `--model ...` (+ bonus `realesrgan-x2plus` cho 2x thật) |
| Engine `realesrgan` (GPU Vulkan) | `--engine realesrgan` (PyTorch CUDA) |
| Engine `cpu_emulation` (SwiftShader) | `--engine cpu` (PyTorch CPU) |
| Engine `ffmpeg_hq` (Lanczos + Unsharp, 3 pass) | `--engine ffmpeg_hq` |
| Video: tách frame → AI → ghép lại + giữ audio | `upscale_video()` (ffmpeg + torch) |
| Batch nhiều file, queue, tự fallback khi AI lỗi | `process_media()` + CLI nhận nhiều file/thư mục/glob |
| Giao diện Electron | `python -m valo_upscale.app` (Gradio, chạy trên Colab/Kaggle) |
| Khoá bản quyền / mã máy | ❌ bỏ (bản tự viết lại, không có license server) |

---

## 1. Cài đặt

Yêu cầu: **Python ≥ 3.9**, **ffmpeg** trong PATH (Colab/Kaggle có sẵn), `torch` (Colab/Kaggle có sẵn bản CUDA).

```bash
git clone https://github.com/hoannguyen171/UPSCALEVIDEO.git
cd valo-upscale-gpu
pip install -r requirements.txt          # numpy, Pillow, requests
pip install -r requirements-ui.txt       # tuỳ chọn: Gradio UI
```

Weight Real-ESRGAN (~2–64 MB) được tải tự động từ GitHub Releases ở lần chạy đầu,
lưu ở `~/.cache/valo-upscale/weights` (đổi bằng biến môi trường `VALO_WEIGHTS_DIR`).

Chạy thử ngay bằng file mẫu có sẵn trong repo:

```bash
python -m valo_upscale.cli samples/sample_lowres.png -s 4
python -m valo_upscale.cli samples/sample_clip.mp4   -s 2
```

---

## 2. Dùng CLI

```bash
# Ảnh, 4x, model mặc định (realesrgan-x4plus)
python -m valo_upscale.cli anh.jpg -s 4

# Video anime, 2x, model tối ưu video
python -m valo_upscale.cli video.mp4 -s 2 -n realesr-animevideov3

# Cả thư mục ảnh, 8x, xuất vào ket_qua/
python -m valo_upscale.cli ./anh_goc -s 8 -o ket_qua

# Không có GPU: Lanczos + Unsharp thuần CPU
python -m valo_upscale.cli video.mp4 -s 6 -e ffmpeg_hq

# GPU ít VRAM: chia tile 400px
python -m valo_upscale.cli anh_4k.jpg -s 4 --tile 400

# Liệt kê model + trạng thái weight
python -m valo_upscale.cli --list-models
```

Bảng tham số chính:

| Tham số | Ý nghĩa |
|---|---|
| `-s, --scale` | `2`, `3`, `4`, `6`, `8` (mặc định `4`) |
| `-n, --model` | `realesrgan-x4plus` (ảnh), `realesrgan-x4plus-anime` (anime/2D), `realesr-animevideov3` (video), `realesrgan-x2plus` |
| `-e, --engine` | `realesrgan` (GPU), `cpu` (CPU), `ffmpeg_hq` (không cần torch) |
| `-t, --tile` | Kích thước tile khi suy luận; `0` = cả ảnh, tự giảm khi hết VRAM |
| `--tile-pad` | Vùng chồng lấn giữa tile (mặc định 10) |
| `--fp32` | Tắt half precision (dùng khi fp16 gây lỗi) |
| `--frame-format` | `jpg` (nhanh, mặc định) hoặc `png` (chất lượng cao nhất) |
| `-o, --outdir` | Thư mục kết quả (mặc định `outputs/`) |

Tên file kết quả trùng quy ước app gốc: `<ten>_upscaled_<scale>x_<timestamp>.png|.mp4`.

### Thư viện

```python
from valo_upscale import process_media

result = process_media("video.mp4", out_dir="outputs", scale=2,
                       model="realesr-animevideov3", engine="realesrgan", tile=400)
print(result["output_path"], result["info"]["width"], "x", result["info"]["height"])
```

### Giao diện web (Gradio)

```bash
pip install -r requirements-ui.txt
python -m valo_upscale.app --share     # --share để lấy link công khai trên Colab/Kaggle
```

---

## 3. Đẩy lên GitHub

```bash
cd valo-upscale-gpu
git init -b main
git add .
git commit -m "valo-upscale-gpu: bản Colab/Kaggle port từ app Valo Upscale"
git remote add origin https://github.com/hoannguyen171/UPSCALEVIDEO.git
git push -u origin main
```

Xác thực khi push (GitHub không nhận mật khẩu thường):

* **Personal Access Token** (dễ nhất): GitHub → *Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token*, tick scope **`repo`**. Khi `git push` hỏi:
  * `Username`: tên GitHub của bạn
  * `Password`: dán **token** vừa tạo
  Lưu lại để không phải nhập lại: `git config --global credential.helper store`
* **GitHub CLI**: `gh auth login` rồi `gh repo create UPSCALEVIDEO --public --source=. --push`
* **GitHub Desktop**: *Add local repository* → *Publish repository*.

Lưu ý cho Colab/Kaggle: **repo phải public** để clone không cần token. Nếu muốn để private,
tạo PAT có scope `repo` và clone bằng:

```bash
git clone https://hoannguyen171:<TOKEN>@github.com/hoannguyen171/UPSCALEVIDEO.git
```

---

## 4. Chạy trên Google Colab (GPU)

1. Mở [colab.research.google.com](https://colab.research.google.com) → *File → Open notebook → GitHub* →
   dán `https://github.com/hoannguyen171/UPSCALEVIDEO` → chọn `colab_upscale.ipynb`
   (hoặc dùng link trực tiếp: `https://colab.research.google.com/github/hoannguyen171/UPSCALEVIDEO/blob/main/colab_upscale.ipynb`).
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU** → *Save*.
3. Sửa ô cấu hình đầu tiên: `REPO_URL` = link repo của bạn (để trống nếu mở notebook từ GitHub).
4. **Runtime → Run all**. Notebook tự: kiểm tra GPU → clone repo → cài deps → chạy thử file mẫu →
   cho bạn upload file / lấy file từ Google Drive → nén kết quả để tải về.

Gói miễn phí Colab dùng GPU T4 16 GB: video 480p→1080p dài vài chục giây mất khoảng 1–3 phút.
Colab sẽ ngắt phiên nếu bạn đóng tab — nhớ tải kết quả về trước.

## 5. Chạy trên Kaggle (GPU)

1. [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → *File → Import Notebook* →
   chọn tab **GitHub** (hoặc upload trực tiếp `kaggle_upscale.ipynb`).
2. Cột phải → **Settings → Accelerator → GPU T4 x2** (hoặc P100) → bật **Internet** (cần để tải weight).
3. Sửa `REPO_URL` trong ô cấu hình → **Run all**.

Kaggle khác Colab ở chỗ:

* Input nằm trong `/kaggle/input/...` (dataset/notebook đã add) — notebook tự quét thư mục này.
* Kết quả ghi vào `/kaggle/working` sẽ được lưu khi bạn *Save version* (dùng để tải về sau).
* Không có `files.download`; tải file bằng tab **Output** của notebook version.

---

## 6. Kiến trúc

```
valo_upscale/
├── archs.py       # RRDBNet + SRVGGNetCompact (thuần torch, không cần basicsr)
├── models.py      # registry model, tải weight, tự suy kiến trúc từ state_dict
├── engine.py      # suy luận có tile/overlap, pipeline ảnh & video, 3-pass fallback
├── dims.py        # clamp 4K (video) / 8K (ảnh), ép cạnh chẵn — port từ bản JS
├── probe.py       # ffprobe: fps, duration, rotation, audio
├── fftools.py     # định vị ffmpeg/ffprobe, chạy subprocess không qua shell
├── remote.py      # đọc thư mục công khai (Drive/GitHub/Dropbox/HTTP) + đẩy kết quả ngược lên
├── cli.py         # dòng lệnh
├── app.py         # UI Gradio (file trên máy)
└── kaggle_app.py  # UI Gradio cho luồng "link thư mục công khai" + chế độ headless
```

Điểm khác biệt kỹ thuật so với bản gốc (đã kiểm chứng khi port):

* Bản gốc gọi `realesrgan-ncnn-vulkan.exe` (Vulkan) — Colab/Kaggle **không có Vulkan**, nên bản này
  dùng PyTorch CUDA với **cùng weight chính thức** (`RealESRGAN_x4plus.pth`, `realesr-animevideov3.pth`, …).
* Model `realesr-animevideov3` trên torch chỉ có bản 4x (SRVGGNetCompact 16 conv), nên mốc 2x/3x chạy 4x
  rồi resample Lanczos — giống cách bản gốc xử lý 6x/8x. Muốn 2x "thuần AI" thì dùng `realesrgan-x2plus`.
* Không còn `basicsr`/`realesrgan` pip (hay vỡ với torchvision mới): kiến trúc được viết lại trong `archs.py`
  và `load_state_dict(strict=False)` + kiểm tra key để đảm bảo nạp đúng weight.

## 7. Sự cố thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `Không tìm thấy ffmpeg` | Colab/Kaggle: `!apt-get install -y ffmpeg` (Kaggle có sẵn). Local: cài ffmpeg hoặc đặt `VALO_FFMPEG=/duong/dan/ffmpeg`. |
| `CUDA out of memory` | Giảm `--tile 400` (hoặc 200), giữ `--tile-pad 10`. |
| Suy luận fp16 ra vệt/nhiễu | Thêm `--fp32`. |
| Video ra quá 4K bị cắt bớt tỉ lệ | Đúng thiết kế: bản gốc clamp 3840×2160 (ngang) / 2160×3840 (dọc) để H.264 giải mã được. |
| Tải weight lỗi/timeout | Chạy lại (có `.part` resume, thử URL dự phòng), hoặc tải thủ công vào `~/.cache/valo-upscale/weights`. |
| Colab báo `numpy` xung đột sau khi cài Gradio | Restart runtime rồi chạy lại từ ô import; CLI không cần Gradio. |

## 8. Giấy phép & ghi công

Code trong repo này: **MIT** (xem `LICENSE`).
Real-ESRGAN/BasicSR: BSD-3-Clause; weight mô hình tải từ GitHub Releases của tác giả gốc
(một số dòng model giới hạn phi thương mại — kiểm tra trước khi dùng cho mục đích thương mại).

Đây là bản **tự viết lại** dựa trên phân tích tĩnh `app.asar` của *Valo Upscale*; repo không chứa
file nhị phân, `config.json` máy chủ, hay bất kỳ thành phần khoá bản quyền nào của bản gốc.

---

## 9. Luồng "link thư mục công khai" (UI trên Kaggle/Colab)

Notebook: **`folder_ui.ipynb`**
* Kaggle: *New Notebook → File → Import Notebook → GitHub* → `hoannguyen171/UPSCALEVIDEO` → `folder_ui.ipynb`
* Colab: https://colab.research.google.com/github/hoannguyen171/UPSCALEVIDEO/blob/main/folder_ui.ipynb

Luồng: dán link thư mục → xem trước danh sách file → tải về → upscale bằng GPU → đẩy kết quả lên đích bạn chọn.
Cài thêm thư viện cho luồng này:

```bash
pip install -q -r requirements-remote.txt   # gdown + google-api-python-client + google-auth
```

### Đọc và ghi được đến đâu (đọc kỹ mục này)

| Nguồn / đích | Đọc bằng link công khai | Đẩy kết quả ngược lên |
|---|---|---|
| Google Drive — thư mục "Anyone with the link" | ✔ (qua `gdown`, không cần đăng nhập) | ✖ cần **Service Account JSON**: tạo SA, bật Drive API, **share thư mục cho `client_email` (quyền Editor)** |
| Dropbox — link chia sẻ | cần **access token** (bấm *Add to your Dropbox* để thư mục nằm trong tài khoản bạn) | ✔ cùng token đó |
| GitHub — repo/thư mục public | ✔ (không cần token) | ✔ cần PAT (`Contents: Read and write`), tối đa 100 MB/file |
| Trang index HTTP / danh sách URL | ✔ | ✖ (không có API ghi) |
| `/kaggle/working` (`kaggle_local`) | — | ✔ luôn dùng được — tải ở tab **Output** sau khi Save version |

**Tóm lại:** link công khai chỉ để *tải về*. Muốn "upscale xong đẩy về đúng thư mục cũ" thì thư mục đó phải
nằm trong tài khoản bạn có quyền ghi (Drive + Service Account, Dropbox + token, hoặc GitHub + PAT).

### UI

```bash
python -m valo_upscale.kaggle_app            # mở UI local
python -m valo_upscale.kaggle_app --share    # tạo link công khai (Kaggle/Colab cần Internet = On)
```

Trong UI có nút **Xem trước danh sách file** (không tải) và **Tải → Upscale → Đẩy lên** (log + tiến trình
theo từng file). Phần *Credential* chỉ cần khi đẩy lên Drive/Dropbox/GitHub; file Service Account JSON
có thể kéo–thả thẳng vào UI.

### Chạy không giao diện (headless)

```bash
export DROPBOX_TOKEN=...      # hoặc GITHUB_TOKEN / GDRIVE_SA_JSON=/duong/dan/sa.json
python -m valo_upscale.kaggle_app --list --url <link>
python -m valo_upscale.kaggle_app --headless --url <link> --scale 4 --limit 5 \
       --upload-kind dropbox --upload-folder /Upscale/ket_qua
```

### Dùng như thư viện

```python
from valo_upscale import remote

items = remote.list_remote("https://drive.google.com/drive/folders/<ID>")   # tự nhận provider
paths = remote.download_remote(items, "input")
remote.upload_results(paths, kind="github", folder="user/repo/ket_qua", github_token="...")
```

---

## 10. Đã kiểm chứng (chạy thật, CPU)

| Hạng mục | Kết quả đo được |
|---|---|
| Nạp weight chính thức | `realesrgan-x4plus` (64 MB), `realesrgan-x4plus-anime` (17 MB, RRDBNet 6 block), `realesr-animevideov3` (2,4 MB, SRVGGNetCompact 16 conv), `realesrgan-x2plus` — `load_state_dict` khớp đủ key, kiến trúc tự suy từ state_dict |
| Ảnh 240×135 → 4x | 960×540 PNG, engine `realesrgan` |
| Ảnh 240×135 → 6x (native 4x + Lanczos/unsharp) | 1440×810 |
| Ảnh 240×135 → 2x bằng `realesrgan-x2plus` (cạnh lẻ 135) | 480×270 — nhờ đệm cạnh cho `pixel_unshuffle` |
| Ảnh PNG có alpha | kênh alpha được giữ nguyên ở đầu ra |
| Video 320×180/12 fps **có audio** → 4x `ffmpeg_hq` | 1280×720, 24 frame, audio còn nguyên |
| Video → 2x qua AI | 640×360, 24 frame, audio còn nguyên, engine `realesrgan` (không fallback) |
| Tiling (`--tile` 64/100/128/200 so với không tile) | lệch tối đa 0,014 / thang 0–1; 100% pixel lệch nằm sát mép tile, tổng < 0,14% pixel |
| Fallback khi AI lỗi (giả lập hết VRAM) | tự chuyển `ffmpeg_hq` và ghi cảnh báo vào log |
| Clamp khung hình | 1920×1080→4x = 3840×2160; 1080×1920→4x = 2160×3840; 3000×2000→8x = 8192×5460 |
| Tên model sai | báo lỗi rõ ràng, exit code 2 (không âm thầm fallback) |
| UI Gradio | gọi end-to-end qua HTTP (`/handler`): trả gallery ảnh + video + file tải về cho cả 2 loại media |
| Luồng thư mục (trang index HTTP local): tải 2 file → upscale 2x → đẩy về thư mục local | 2/2 file thành công, video giữ nguyên audio |
| Luồng thư mục + engine AI | clip 160×90 qua `realesrgan` ra video kết quả, không fallback |
| GitHub thật (`hoannguyen171/UPSCALEVIDEO/tree/main/samples`) | liệt kê 3 file, tải `sample_clip.mp4` (86 587 byte) từ raw.githubusercontent |
| GitHub (API giả lập): lọc media theo thư mục con + upload | chỉ lấy `a.mp4`, `b.png`; PUT contents trả 201, đúng nội dung |
| Dropbox (API giả lập): list + download + upload | chỉ lấy media, tải đúng bytes, upload đúng path `/Upscale/ket_qua/...` |
| Drive (giả lập gdown + service): list/download/upload | lọc media + `file_id`, upload đúng `parents`, mimetype `video/*`, trả `webViewLink`; ID sai → báo lỗi rõ |
| UI Gradio (gọi qua HTTP `/preview` + `/handler`) | xem trước 3 file; chạy 1 file ra kết quả, log "Hoàn tất" |
| CLI headless (`--list`, `--headless --limit 1`) | rc = 0, file nằm trong `upscaled/` |
| `colab_upscale.ipynb`, `kaggle_upscale.ipynb`, `folder_ui.ipynb` | JSON hợp lệ, mọi code cell compile được |

Kiến trúc mạng được đối chiếu trực tiếp với mã nguồn gốc `BasicSR/basicsr/archs/rrdbnet_arch.py`
và `Real-ESRGAN/realesrgan/archs/srvgg_arch.py`.

Lưu ý: `realesrgan-ncnn-vulkan.exe` của app gốc **không chạy được** trên máy không có Vulkan
(`vkCreateInstance failed -9`) — đúng lý do bản port này dùng PyTorch thay vì ncnn.

Chưa chạy thật với endpoint Google Drive / Dropbox (máy này không có credential): các nhánh đó đã được
kiểm tới tầng HTTP/API (server giả lập nhận đúng request) và tầng gọi thư viện (`gdown`, `MediaFileUpload`),
nhưng lần chạy thật đầu tiên trên Kaggle vẫn nên thử với 1 file nhỏ.
