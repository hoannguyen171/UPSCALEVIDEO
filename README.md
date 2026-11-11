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
| (bổ sung) Upscale video bằng RelayVSR — model diffusion của Wan 2.2 | notebook `relayvsr_kaggle.ipynb` + `python -m valo_upscale.comfy_upscale` (chạy trong ComfyUI) |
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
| `--devices` | `auto` (mặc định — dùng hết GPU) · `0` · `0,1` · `cpu` |
| `--batch` | Số patch/frame mỗi lần forward (mặc định 4; chỉ áp dụng cho GPU, CPU tự về 1) |
| `--encoder` | `auto` (NVENC nếu có) · `nvenc` · `x264` |
| `--no-hwaccel` | Tắt giải mã NVDEC |
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
├── engine.py      # suy luận đa GPU + batch + pipeline 3 tầng, ảnh/video, NVENC/NVDEC, 3-pass fallback
├── dims.py        # clamp 4K (video) / 8K (ảnh), ép cạnh chẵn — port từ bản JS
├── probe.py       # ffprobe: fps, duration, rotation, audio
├── fftools.py     # định vị ffmpeg/ffprobe, chạy subprocess không qua shell, probe NVENC/NVDEC
├── remote.py      # đọc thư mục công khai (Drive/GitHub/Dropbox/HTTP) + đẩy kết quả ngược lên
├── lightx2v_upscale.py  # đường LightX2V: 2 GPU cho 1 clip (Ulysses SP), tải weight, hậu kiểm + ghép audio
├── cli.py         # dòng lệnh
├── app.py         # UI Gradio (file trên máy)
├── kaggle_app.py  # UI Gradio cho luồng "link thư mục công khai" + chế độ headless
└── comfy_upscale.py  # client ComfyUI headless: chạy RelayVSR (video SR diffusion), tải weight, dựng workflow
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

Phần chạy trên GPU (Kaggle) dùng thêm các thành phần bên thứ ba — repo **không** chứa chúng, worker tự tải lúc chạy:

| Thành phần | Giấy phép |
|---|---|
| ComfyUI | GPL-3.0 |
| [ComfyUI-ReCaVSR](https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR) | **GPL-3.0** (do nó dùng lại mã của ComfyUI) |
| Mã ReCaVSR (ĐH KH&CN Trung Quốc) nằm trong node pack: kernel Triton attention, FlashDecoder, cache KV, DiT/session | Apache-2.0 (`LICENSE-APACHE` của node pack) |
| Weight `PedroMarinhoDev/ReCaVSR-ComfyUI` (int8), `kopper/ReCaVSR` (prompt/flashdecoder/bf16), `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (VAE) | theo điều kiện của tác giả từng repo |

⚠️ Nếu bạn **phát hành một gói có kèm** ComfyUI/node pack (ví dụ đóng gói chung vào app), nghĩa vụ của
**GPL-3.0** áp dụng cho phần đó — còn repo này (MIT) chỉ gọi chúng qua mạng.

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

## 10. Tăng tốc (đa GPU, batch, NVENC/NVDEC)

"GPU chỉ chạy 30%" trên Kaggle T4 x2 đến từ 3 nguyên nhân: chỉ dùng **1 trong 2 GPU**, mỗi lần forward chỉ
**1 patch/1 frame** (GPU đói), và **CPU** (giải mã/ghi JPEG, encode H.264) chặn giữa các lần forward.

| Tối ưu | Cơ chế | Cách điều khiển |
|---|---|---|
| **Nhiều GPU** | mỗi GPU một model riêng, chia patch (ảnh) / chia khối frame (video) | `--devices auto` (mặc định, dùng hết GPU) · `0` · `0,1` · `cpu` |
| **Batch** | gộp nhiều patch/frame vào 1 lần forward | `--batch 4` (mặc định trên GPU; CPU tự ép về 1) |
| **Pipeline 3 tầng** | luồng đọc frame ‖ luồng suy luận (1/GPU) ‖ luồng ghi frame | tự động |
| **NVENC / NVDEC** | encode/giải mã video bằng phần cứng | `--encoder auto` (mặc định) · `x264` · `--no-hwaccel` |
| **channels_last + cudnn.benchmark + truyền uint8 lên GPU** | giảm băng thông, conv nhanh hơn | tự động trên CUDA |
| **Cache model** | nạp model 1 lần cho cả batch/queue | tự động |
| **Tự hạ tải khi hết VRAM** | giảm batch dần rồi mới bật tile, thay vì crash | tự động |

### Số đo (máy phát triển này chỉ có CPU, không có GPU NVIDIA)

| Phép đo | Bản cũ | Bản này | Kết quả |
|---|---|---|---|
| Pipeline đọc/suy luận/ghi (compute giả lập 25 ms/frame, 40 frame 1280×720) | 2,80 s (14,3 fps) | **1,36 s (29,4 fps)** | **nhanh hơn 2,06×** |
| Ảnh 4x tile 400, 1280×720 (4 lần chạy, lấy trung vị) | 17,0 s | 16,7 s | tương đương (CPU bão hoà sẵn, nhiễu ±25%) |
| Video 720p 2x, 30 frame | 472,6 s | 464,7 s | tương đương — CPU mất 15,7 s/frame nên I/O không đáng kể |
| 3 file qua `process_media` (nhờ cache model) | 3,97 s | 3,71 s | nhanh hơn 7% (weight nhỏ; RRDB x4plus lợi hơn) |

**Chưa đo được trên GPU thật**: máy này không có NVIDIA, nên phần nhiều GPU / batch / NVENC / NVDEC chỉ kiểm được
bằng worker giả (chia việc đúng, giữ nguyên thứ tự) và probe ffmpeg (trả `False` chính xác khi không có GPU).
Ước lượng trên Kaggle **T4 x2** [INFERENCE]: dùng đủ 2 GPU (~1,8×) + GPU không còn chờ I/O (phần đo được là 2,06×)
+ batch tăng mức sử dụng SM + NVENC/NVDEC gỡ nút thắt CPU → **nhanh hơn khoảng 2–4×** so với bản trước,
và `nvidia-smi` sẽ thấy util tăng rõ.

### Cách tự kiểm tra trên Kaggle

```bash
# xem mức sử dụng GPU trong lúc chạy (terminal khác)
watch -n 1 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

```python
# quét batch để chọn giá trị tốt nhất cho clip của bạn (log in ra fps từng khối frame)
for b in (1, 2, 4, 6, 8):
    !python -m valo_upscale.cli samples/sample_clip.mp4 -s 2 --batch {b}
```

Nếu VRAM không đủ: tự động hạ batch/bật tile, hoặc đặt tay `--tile 400`. Nếu input là codec lạ làm NVDEC lỗi:
`--no-hwaccel`. Nếu cần tương thích tối đa với player cũ: `--encoder x264`.

---

## 11. Đã kiểm chứng (chạy thật, CPU)

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
| Engine tối ưu — suy luận thuần (cũ vs mới) | lệch tối đa 1/255, chỉ 49 trên 44 triệu pixel |
| Engine tối ưu — `tile=0` (cũ vs mới) | lệch tối đa 2/255 (làm tròn uint8) |
| Engine tối ưu — `batch 1` vs `batch 4` | giống hệt (0/255) |
| Engine tối ưu — frame AI trung gian của video | lệch tối đa 2/255 (187/2,7 triệu pixel) |
| Engine tối ưu — ảnh có tile (cũ vs mới) | PSNR 64 dB (khác cách đặt patch ở mép tile) |
| Chia patch cho 2 GPU (worker giả) | giữ đúng thứ tự, chia 2/1 patch, đủ 3 ảnh |
| Tự hạ tải khi "hết VRAM" (giả lập OOM) | vẫn ra kết quả 256×256 đúng, batch tự về 1 |
| Probe NVENC/NVDEC | trả `False` đúng trên máy không có GPU (probe NVDEC phải ép `hwdownload` mới chính xác) |
| Pipeline 3 tầng | nhanh hơn 2,06× khi compute bằng nhau (mục 10) |
| RelayVSR — node & input của ComfyUI | ComfyUI master `1332fab` (06/11/2026): đủ 9 node, tên input khớp graph dựng bằng code |
| RelayVSR — workflow hợp lệ | chạy tới node `UNETLoader` rồi mới lỗi khi nạp weight giả (`Error while deserializing header`) — không có lỗi schema |
| RelayVSR — client + node video core (graph stub, trên server thật) | `LoadVideo→GetVideoComponents→CreateVideo→SaveVideo` ra mp4 240×136/10 fps, **giữ audio**, client copy được file ra ngoài |
| RelayVSR — `upscale_video()` end-to-end (graph stub) | trả file kết quả, giữ kích thước + audio, dọn file tạm trong `input/`, báo đúng `expected_size` (2x) |
| RelayVSR — suy luận thật | **chưa kiểm chứng** (máy dev không có GPU) — phải chạy trên Kaggle |
| LightX2V — config & lệnh chạy | 6/6 config chính thức có thật (x2/x4 × `single`/`sp`/`block_offload`), `sp` đúng `{"seq_p_size": 2, "seq_p_attn_type": "ulysses"}`; lệnh dựng khớp `scripts/dist_infer/*.sh` (`torch.distributed.run --nproc_per_node=N -m lightx2v.infer …`) |
| LightX2V — weight | đúng repo + đúng tên file cho cả x2 (`lightx2v/RelayVSR_X2`) và x4 (`kopper/RelayVSR`), mọi lần gọi đều có token, dùng lại bản sẵn có (symlink, không gọi mạng), file <1 MB bị coi là chưa sẵn sàng; **4/4 file tồn tại thật trên Hub** (HfApi) |
| LightX2V — đường ống end-to-end | chạy bằng repo LightX2V **giả**: subprocess thật → mp4 đúng 2×, **ghép lại audio** từ video gốc, xoá file trung gian, giữ log; `size_ok`/`audio_ok`/`frames_ok` đều True; `dry-run` không thực thi gì |
| LightX2V — hậu kiểm & audio | `expected_size` làm tròn chẵn đúng (1280×720 → 2560×1440; 101×51 → 202×102); `ensure_audio` thật bằng ffmpeg (ghép/ bỏ qua khi đã có tiếng/ khi nguồn không có tiếng) |
| LightX2V — suy luận thật | **chưa kiểm chứng** (máy dev không có GPU) — phải chạy trên Kaggle |
| RelayVSR — tải weight: token, `source_dir`, symlink, bỏ qua file đã có | 15/15 PASS (`_smoke/test_hf_weights.py`): token đọc từ tham số/env, `hf_hub_download` nhận token, file vào đúng `models/diffusion_models` + `models/vae`, có sẵn thì symlink và **không gọi mạng** |
| RelayVSR — token HF thật | `whoami` OK (user `hoanvg17195`), repo public truy cập được |
| RelayVSR — 2 GPU cho nhiều clip | 2 instance ComfyUI: port riêng, `input/output/temp/user` riêng, **chung `models`**, `check_ready` OK trên cả hai, 4 clip chạy **chồng thời gian** (chứng minh song song), kết quả đúng thứ tự + giữ audio, dọn sạch file tạm |
| RelayVSR — 1 instance vẫn giữ luồng cũ | `start_server`/`start_instance(isolate=False)` dùng nguyên `input`/`output` mặc định của ComfyUI; chỉ `start_instances` mới tách thư mục (test cũ chạy lại: 24/24 PASS, không hồi quy) |
| Liệt kê kết quả trong notebook | Ô "Liệt kê / nén kết quả" quét **cả thư mục con** (`rglob`) — trước đây ô 6b ghi vào `<OUT_DIR>/song_song/` nên ô liệt kê (chỉ quét 1 cấp) báo 0 file; test mô phỏng đúng cây thư mục đó: 3 file/2 thư mục, zip chứa cả file lồng, không crash ngoài Colab |
| RelayVSR — chống treo khi ComfyUI chết lúc khởi động | `wait_ready(..., proc=, log_path=)` phát hiện tiến trình thoát và in log cuối (thay vì chờ hết 900s) |
| RelayVSR — tải cả x2 và x4 | 25/25 PASS (`_smoke/test_hf_weights.py`): `normalize_scales` nhận `2` / `4` / `"2,4"` / list (bỏ trùng, giữ thứ tự), hệ số sai bị chặn; tải cả 2 cặp = **4 file đúng tên**, mỗi lần gọi đều mang token |
| RelayVSR — tên file trên Hub | 4/4 file (x2 + x4, model + VAE) tồn tại thật trên `Comfy-Org/RelayVSR` (kiểm qua HfApi) |

Kiến trúc mạng được đối chiếu trực tiếp với mã nguồn gốc `BasicSR/basicsr/archs/rrdbnet_arch.py`
và `Real-ESRGAN/realesrgan/archs/srvgg_arch.py`.

Lưu ý: `realesrgan-ncnn-vulkan.exe` của app gốc **không chạy được** trên máy không có Vulkan
(`vkCreateInstance failed -9`) — đúng lý do bản port này dùng PyTorch thay vì ncnn.

Chưa chạy thật với endpoint Google Drive / Dropbox (máy này không có credential): các nhánh đó đã được
kiểm tới tầng HTTP/API (server giả lập nhận đúng request) và tầng gọi thư viện (`gdown`, `MediaFileUpload`),
nhưng lần chạy thật đầu tiên trên Kaggle vẫn nên thử với 1 file nhỏ.

## 12. RelayVSR — upscale video bằng model diffusion (qua ComfyUI)

[Comfy-Org/RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR) là bản int8 đóng gói cho ComfyUI của
[kopper/RelayVSR](https://huggingface.co/kopper/RelayVSR) — **video super-resolution** kiểu diffusion:
chỉ **mỗi frame thứ 15** đi qua model Wan 2.2 (sparse keyframes), phần còn lại do decoder "flash" tái tạo.
Chạy **1 step, CFG 1.0**, và giữ nguyên audio của clip gốc.

Hệ số phóng đại **do checkpoint quyết định** (2× hoặc 4×), không phải tham số như Real-ESRGAN.

| | Real-ESRGAN (mục 2) | RelayVSR |
|---|---|---|
| Cách chạy | từng frame, PyTorch trong repo này | graph ComfyUI, 1 step |
| Hệ số | 2/3/4/6/8 tuỳ chọn | cố định 2× hoặc 4× theo checkpoint |
| Audio | ffmpeg ghép lại sau | truyền thẳng trong graph (giữ nguyên, không remux) |
| Điểm mạnh | nhanh, upscale hàng loạt | thêm chi tiết "sinh" mới, **giảm nhấp nháy theo thời gian** |
| Tốc độ tham chiếu | T4: vài fps/frame | H100: 45,4 fps cho 1024×768→2048×1536 (x2); 12,6 fps cho 4× output |
| Khi nào dùng | mọi lúc, nhất là batch lớn | clip AI bị flicker, cần bản nét hơn hẳn |

### Token Hugging Face (tuỳ chọn) & không tải lại 5,3 GB

* Repo `Comfy-Org/RelayVSR` **công khai** — không cần token; nhưng token giúp **hạn mức/tốc độ tải** cao hơn:

```bash
export HF_TOKEN=hf_xxx          # hoặc
python -m valo_upscale.comfy_upscale --download --comfy-dir ComfyUI --scales 2 --token hf_xxx
python -m valo_upscale.comfy_upscale --download --comfy-dir ComfyUI --scales "2,4"      # tải cả hai cặp
```

```python
cu.download_models("/kaggle/working/ComfyUI", 2, token="hf_xxx")        # chỉ x2
cu.download_models("/kaggle/working/ComfyUI", "2,4", token="hf_xxx")    # cả x2 và x4 (~10,6 GB)
```

* Đã có weight ở đâu đó (thường là `/kaggle/input` của một Kaggle Dataset) thì truyền `source_dir`:
  hàm sẽ **symlink** (không tốn thêm 5,3 GB đĩa) hoặc copy nếu không symlink được:

```python
cu.download_models("/kaggle/working/ComfyUI", 2, source_dir="/kaggle/input")
```

Quy trình tiết kiệm nhất trên Kaggle: phiên đầu tải 1 lần → **Save version** → tab **Output** →
*New Dataset* từ `ComfyUI/models`; các phiên sau *Add Input* dataset đó và để `WEIGHTS_SOURCE="/kaggle/input"`.
Nên lưu token trong **Kaggle Secrets** (Add-ons → Secrets, tên `HF_TOKEN`) thay vì gõ vào notebook.

### Yêu cầu

* **ComfyUI master sau 06/11/2026** (PR #17511 của kijai) — bản release cũ **chưa có** node
  `RelayVSRConditioning` và `VSRDecode`. Notebook tự clone master và kiểm tra.
* Weight int8 (Comfy-Org), **không trộn x2 với x4**:

| Scale | File model → `models/diffusion_models/` | File VAE → `models/vae/` |
|---|---|---|
| 2× | `relayvsr_x2_lightx2v_int8_convrot.safetensors` (5,16 GB) | `relayvsr_flash_vae_x2_fp32.safetensors` (122 MB) |
| 4× | `relayvsr_x4_int8_convrot.safetensors` (5,16 GB) | `relayvsr_flash_vae_x4_fp32.safetensors` (122 MB) |

* Kaggle: *Accelerator = GPU T4 x2*, **Internet = On**, trống ≥ 6 GB đĩa. T4 16 GB dùng được với bản int8
  (nếu OOM thì thêm `--lowvram`, hoặc dùng cấu hình block-offload của LightX2V).

### Chạy

Notebook: **`relayvsr_kaggle.ipynb`** — ô 3 có `DOWNLOAD_SCALES` (`"2"`, `"4"` hoặc `"2,4"`) và `SCALE`
(hệ số dùng khi chạy, phải nằm trong danh sách đã tải) → clone ComfyUI → tải weight → chạy server headless → upscale →
hiển thị kết quả (nhận cả đường dẫn local, `/kaggle/input`, hoặc **link thư mục công khai** qua `remote.py`).

Bằng CLI:

```bash
python -m valo_upscale.comfy_upscale --download --comfy-dir /kaggle/working/ComfyUI --scale 2
python -m valo_upscale.comfy_upscale --check    --comfy-dir /kaggle/working/ComfyUI --scale 2
python -m valo_upscale.comfy_upscale --input clip.mp4 --out ket_qua.mp4 \
       --comfy-dir /kaggle/working/ComfyUI --scale 2
```

Bằng thư viện:

```python
from valo_upscale import comfy_upscale as cu

cu.wait_ready("http://127.0.0.1:8188")
result = cu.upscale_video("clip.mp4", "out.mp4", comfy_dir="/kaggle/working/ComfyUI", scale=2,
                          detail=1.0, progress=lambda sec, state: print(sec, state))
print(result["info"], result["size_ok"])
```

Server có thể tự bật từ Python (đúng như notebook làm):

```python
proc = cu.start_server("/kaggle/working/ComfyUI", port=8188, extra_args=("--lowvram",))
cu.wait_ready("http://127.0.0.1:8188", timeout=900)
```

### Graph dựng bằng code (thuần node core)

```
LoadVideo → GetVideoComponents ─┬→ RelayVSRConditioning → KSampler(1 step, cfg 1) → VSRDecode
                                └→ (images, audio, fps) ──────────────→ CreateVideo → SaveVideo
```

* `RelayVSRConditioning(model, images)` → `positive` + `latent` rỗng; `VSRDecode(vae, samples, images, detail)`
  → ảnh đã upscale (`detail` 0→1: giữ bao nhiêu phần chi tiết model thêm vào bản bicubic).
* `comfy_upscale.build_relayvsr_workflow()` dựng graph; tham số `steps=1`, `cfg=1.0`,
  `sampler="euler"`, `scheduler="simple"`, `detail=1.0` — **chưa có template chính thức** cho model này
  (repo `Comfy-Org/workflow_templates` chưa có file nào), tài liệu của ComfyUI chỉ nói "1 step, cfg 1";
  với 1 step thì chọn sampler/scheduler gần như không đổi kết quả, và có thể đổi qua tham số.
* Vài chi tiết dễ sai, đã phát hiện khi viết client: input `format` của `SaveVideo` là **DynamicCombo** nên
  phải truyền **chuỗi** `"mp4"` (truyền dict → node bị gọi thiếu tham số `format`); history của ComfyUI còn
  chứa một mục `type="input"` (file gốc do `LoadVideo` trả về) nên phải lọc khi lấy kết quả; và trên Windows
  file tạm trong `input/` có thể còn bị giữ handle → phải thử xoá lại vài lần.

### Dùng cả 2 GPU (Kaggle T4 x2)

Hai việc khác nhau, đừng lẫn:

**A. Nhiều clip → tăng thông lượng (đã code + đã test, không cần cài thêm)**

Mỗi GPU một instance ComfyUI: **dùng chung thư mục `models`** nhưng tách `input`/`output`/`temp`/`user` riêng
(`--base-directory` + `--input-directory` + `--output-directory` + `--temp-directory` + `--user-directory`),
rồi chia clip theo vòng:

```python
from valo_upscale import comfy_upscale as cu

instances = cu.start_instances("/kaggle/working/ComfyUI", ports=[8188, 8189], gpus=[0, 1],
                               root="/kaggle/working/comfy_instances")   # nhớ stop_instances() khi xong

outcome = cu.run_clips_parallel(                                          # tự bật, chia clip, tự dọn
    clips, "out", comfy_dir="/kaggle/working/ComfyUI", gpus=[0, 1], scale=2,
    root="/kaggle/working/comfy_instances", on_log=print,
)
print(outcome["wall"], outcome["errors"])
```

Notebook: ô **6b**. Đây là tăng **thông lượng** khi có nhiều clip (~2× với 2 GPU); **không** làm một clip
chạy nhanh hơn. Muốn một clip nhanh hơn thì phải chia chính nó ra — xem B.

**B. Một clip, dùng cả 2 GPU → LightX2V + Ulysses sequence parallel (chưa đóng gói vào repo)**

`configs/relayvsr/relayvsr_x2_sp.json` chỉ khác config thường ở chỗ thêm:

```json
"parallel": { "seq_p_size": 2, "seq_p_attn_type": "ulysses" }
```

→ chuỗi token của DiT được **chia đôi cho 2 GPU**; ở attention, các rank đổi vai trò head↔sequence bằng
`all2all` rồi `all_gather` (`lightx2v/models/networks/relayvsr/parallel.py`). Khởi động bằng **`torchrun`**,
số process = `seq_p_size × cfg_p_size` (docs `method_tutorials/parallel.md`):

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DTYPE=FP16        # T4 (Turing) không có BF16 native, mặc định của LightX2V là BF16
torchrun --nproc_per_node=2 -m lightx2v.infer \
  --model_cls relayvsr_x2 --task sr \
  --model_path ./models/RelayVSR_X2 \
  --config_json ./configs/relayvsr/relayvsr_x2_sp.json \
  --video_path input.mp4 --sr_ratio 2 --save_result_path save_results/out.mp4 --seed 42
```

Đường này dùng bản **fp16** của LightX2V (`wan_x2.safetensors` 9,66 GB + `flash_x2.safetensors` 122 MB, repo
`lightx2v/RelayVSR_X2`), **không** dùng file int8 của Comfy-Org. Bản 4x có `relayvsr_x4_sp.json` với
`wan.safetensors`/`flash.safetensors`.

Rủi ro cần biết trước: LightX2V là framework production — `requirements.txt` gồm `sgl-kernel`, `qtorch`,
`torchao`, `torchaudio`, `decord`, `redis`, `pymongo`, `aioboto3`, `swanlab`, `modelscope`…; tài liệu khuyến
nghị Docker `lightx2v/lightx2v:26062001-cu130` (Kaggle không chạy Docker được). Config `_sp` bật
`use_compile: true` + `attention_backend: triton` nên lần chạy đầu phải biên dịch.

**Đã đóng gói** đường này: module `valo_upscale/lightx2v_upscale.py` + notebook `relayvsr_lightx2v_kaggle.ipynb`.

```bash
python -m valo_upscale.lightx2v_upscale --check    --repo-dir LightX2V --scale 2
python -m valo_upscale.lightx2v_upscale --download --repo-dir LightX2V --scale 2 --weights-source /kaggle/input
python -m valo_upscale.lightx2v_upscale --dry-run  --input clip.mp4 --out out.mp4 --repo-dir LightX2V --mode sp --nproc 2
python -m valo_upscale.lightx2v_upscale --input clip.mp4 --out out.mp4 --repo-dir LightX2V --mode sp --nproc 2
```

```python
from valo_upscale import lightx2v_upscale as lx

lx.clone_repo("LightX2V", log=print)
for cmd in lx.install_commands("LightX2V"):        # pip install -e . (bỏ qua torch để giữ bản CUDA của Kaggle)
    subprocess.run(cmd, check=False)
lx.download_weights(2, "LightX2V/models/RelayVSR_X2", source_dir="/kaggle/input")

rep = lx.run_upscale("clip.mp4", "out.mp4", repo_dir="LightX2V", scale=2, mode="sp", nproc=2,
                     dtype="FP16", progress=print)
print(rep["size_ok"], rep["audio_ok"], rep["frames_ok"], rep["info"])
```

| `mode` | Config LightX2V | Dùng khi |
|---|---|---|
| `sp` | `relayvsr_x2_sp.json` | **2 GPU cho 1 clip** (Ulysses, `seq_p_size=2`) |
| `single` | `relayvsr_x2.json` | 1 GPU |
| `block_offload` | `relayvsr_x2_block_offload.json` | 1 GPU ít VRAM (`cpu_offload` theo block) |

Weight: x2 → repo `lightx2v/RelayVSR_X2` (`wan_x2.safetensors` 9,66 GB + `flash_x2.safetensors` 122 MB);
x4 → repo `kopper/RelayVSR` (`wan.safetensors` + `flash.safetensors`). Runner SR của LightX2V **không** giữ
audio → module tự ghép lại từ video gốc (`ensure_audio`) rồi hậu kiểm kích thước/fps/số frame
(`verify_output`, có xét việc làm tròn xuống bội số 2 của YUV420).

Mặc định `dtype="FP16"` vì **T4 (Turing) không có BF16 native**, còn `scripts/base/base.sh` của LightX2V để
`DTYPE=BF16`. Cần đổi thêm thì dùng `overrides=`: `{"use_compile": False}` khi torch.compile lỗi,
`{"video_codec": "libx264"}` nếu muốn player/browser đọc được ngay (config gốc là `libx265`).

### Trục trặc thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `check_ready` báo thiếu node | Cập nhật ComfyUI lên master (ô 2 của notebook) |
| OOM trên T4 | `extra_args=("--lowvram",)`, hoặc chia clip thành đoạn ngắn |
| Lỗi khi nạp model int8 trên Turing | Đổi sang weight fp của `kopper/RelayVSR` (`wan.safetensors` 9,6 GB + `flash.safetensors` 122 MB) và trỏ `model_file`/`vae_file` sang tên file đó |
| Dùng cả 2 GPU cho **nhiều clip** | Mục "Dùng cả 2 GPU" bên dưới — 2 instance ComfyUI chia clip, không cần cài thêm |
| Dùng cả 2 GPU cho **một clip** | LightX2V + `relayvsr_x2_sp.json` (Ulysses sequence parallel) — xem cùng mục |
| Thiếu VRAM trên 1 GPU | `relayvsr_x2_block_offload.json` (`cpu_offload` theo block) hoặc `LOW_VRAM` trong notebook |
| Muốn UI trực quan thay vì headless | Custom node cộng đồng `bbaudio-2025/Comfyui-Relay-VSR` (có node "Upscale Streaming" pixel-in/pixel-out) |

Link tham khảo: [model card gốc](https://huggingface.co/kopper/RelayVSR) ·
[RelayVSR X2 của LightX2V](https://huggingface.co/lightx2v/RelayVSR_X2) ·
[hướng dẫn LightX2V](https://github.com/ModelTC/LightX2V/tree/main/scripts/relayvsr) ·
[wiki ComfyUI về RelayVSR X2](https://comfyui-wiki.com/en/news/2026-10-29-relayvsr-x2)

---

## 13. Ứng dụng Windows + GPU Kaggle — ReCaVSR **int8** (4× video)

Model: [ReCaVSR](https://github.com/kopperx/ReCaVSR) (USTC) — one-step streaming video super-resolution trên
Wan2.2 TI2V-5B, chạy qua node pack [PedroMarinhoDev/ComfyUI-ReCaVSR](https://github.com/PedroMarinhoDev/ComfyUI-ReCaVSR).
Dùng bản **int8 ConvRot (6,27 GB, nhanh ~1,8× so với bf16, nhẹ VRAM hơn ~4 GB)** — int8 cần GPU compute
capability ≥ 7.5, **T4 của Kaggle đúng ngưỡng**.

Ứng dụng chạy trên Windows, GPU thì mượn của Kaggle:

```mermaid
graph LR
  A[App Windows<br/>Tkinter] -->|1. mở HTTP server + tunnel tạm| T[cloudflared quick tunnel]
  T -->|2. URL công khai của video| K[Kernel Kaggle<br/>GPU T4]
  K -->|3. ComfyUI + ReCaVSR int8| K
  A -->|4. kaggle kernels push/status| K
  K -->|5. kaggle kernels output| A
```

Kaggle **không** cho kết nối vào notebook, nên chiều đưa video đi dùng tunnel tạm (cloudflared, app tự tải ~53 MB,
không cần tài khoản); chiều lấy kết quả về dùng API chính thức `kaggle kernels output`.

**Hàng đợi GPU làm tunnel chết**: hàng đợi T4 có lúc dài (đã gặp **14 phút**), và tunnel quick của cloudflared
có thể chết trong lúc đó → kernel không tải được video (`HTTP Error 530`). App tự phát hiện (đọc lỗi trong
`valo_result.json`/log worker), **mở tunnel mới rồi đẩy lại kernel** — tối đa 2 lần, có ghi rõ trong nhật ký.

**Dừng tunnel đúng cách**: trên Windows nếu `cloudflared` là shim `.cmd`/`.bat` thì tiến trình thật là con của nó,
nên `winapp/tunnel.py::QuickTunnel.stop()` giết **cả cây** (`taskkill /T /F`) và đóng file log — đã kiểm: sau khi
chạy xong bộ test, **0 tiến trình** cloudflared còn sót (trước khi sửa: 9 tiến trình).

### Chạy bằng 1 cú nhấp

| Cách | File | Ghi chú |
|---|---|---|
| Nhấp đúp | **`Chay_App.cmd`** | tự tìm `pythonw` (không hiện cửa sổ đen); dùng ngay, không cần build |
| Nhấp đúp | `Chay_App.pyw` | nếu Windows đã liên kết `.pyw` với Python |
| **Một file .exe** | chạy **`build_exe.cmd`** một lần → `dist\ValoUpscale.exe` | nhấp đúp là mở app; máy khác không cần cài Python (vẫn cần `kaggle` CLI + `kaggle.json`) |

⚠️ **Sửa `kaggle_jobs/recavsr_worker.py` thì phải build lại exe** — bản exe nhúng worker vào gói, còn bản mã nguồn
đọc worker trong repo (`winapp/job.py::worker_template_path`).

`build_exe.cmd` cài `pyinstaller`, nhúng luôn `kaggle_jobs/recavsr_worker.py` vào gói và tạo **một file duy nhất**
`dist\ValoUpscale.exe` — **~13 MB**, build ~30 s (đã loại numpy/PIL/torch để exe nhẹ và tránh lỗi đệ quy
khi đóng gói; `kagglesdk` thì **có** nhúng sẵn để app chỉ tải file json/log/mp4 thay vì cả cây output). Vì vậy 2 nút *Kiểm tra ComfyUI ở máy* và *Chạy local* trong bản exe sẽ báo cần chạy từ mã nguồn;
muốn có cả 2 nút đó thì chạy `python run_win.py` (máy đã có sẵn numpy/PIL) — toàn bộ đường **GPU Kaggle vẫn đầy đủ trong exe**. Bản exe không có console;
muốn tự kiểm bản đóng gói thì chạy `dist\ValoUpscale.exe --self-test` — nó ghi `valo_self_test.json` cạnh file exe
(đường dẫn worker nhúng, phiên bản tkinter, trạng thái `frozen`). Chạy từ mã nguồn thì `python run_win.py --self-test`
in thẳng ra màn hình. Kết quả thật của `dist\ValoUpscale.exe --self-test`: `frozen: true`, worker nhúng ở
`…\_MEIxxxx\kaggle_jobs\recavsr_worker.py`, `worker_template_exists: true`, `kagglesdk: true`, và đường dẫn `kaggle` CLI.

Muốn kiểm **worker trong exe có khớp repo không** (ví dụ sau khi sửa worker mà quên build lại):

```bash
dist\ValoUpscale.exe --dump-worker worker_tu_exe.py
python -c "import hashlib,pathlib as p; print(hashlib.sha256(p.Path('kaggle_jobs/recavsr_worker.py').read_bytes()).hexdigest())"
python -c "import hashlib,pathlib as p; print(hashlib.sha256(p.Path('worker_tu_exe.py').read_bytes()).hexdigest())"
```

Hai mã sha256 phải giống nhau (đã kiểm thật: `ddf9704bed384ac2…`, 13 154 byte).

### Cài & dùng

```bash
pip install -r requirements-winapp.txt          # chỉ cần CLI kaggle
# Kaggle → Account → Create New API Token → %USERPROFILE%\.kaggle\kaggle.json
python -m winapp.cli --check                    # kiểm CLI + credentials
python -m winapp.cli --tunnel-test              # tự kiểm tunnel thật (tải cloudflared + thử /health + tải thử file)
python run_win.py                               # mở ứng dụng
```

Trong app: chọn video → chọn thư mục kết quả → *Xem trước (dry-run)* để xem kế hoạch → **BẮT ĐẦU (GPU Kaggle)**.

Thời gian thật của một job 2× thành công trên T4 (đọc từ `valo_run.log`, tổng **163,8 s**):

| Giai đoạn | Thời gian |
|---|---|
| Nạp torch/CUDA, clone repo, tải video qua tunnel | 8 s |
| Clone ComfyUI + pip (requirements, triton, opencv/hf) + node pack | 42 s |
| Tải weight int8 + prompt + flashdecoder (6,5 GB) | 14 s (đường truyền Kaggle ~460 MB/s) |
| Khởi động ComfyUI đến khi sẵn sàng (nạp model) | 61 s |
| **Suy luận 24 frame 320×180 → 640×360** | **36,8 s** (≈1,5 s/frame) |
| Dọn dẹp + ghi kết quả | 2 s |

⇒ **Chi phí cố định ~2 phút/job** không phụ thuộc độ dài video, cộng ~1,5 s mỗi frame ở 2× với clip nhỏ.
Số frame tăng theo độ phân giải đầu ra (mức 4× = 4× diện tích pixel → chậm hơn nhiều lần) [ước lượng, chưa đo
ở 4×]. Vì vậy: chạy 1 clip dài rẻ hơn hẳn nhiều clip ngắn.
Nhật ký chạy ngay trong cửa sổ; xong có nút mở video/thư mục kết quả. Hoặc dùng dòng lệnh:

```bash
python -m winapp.cli --video clip.mp4 --out ket_qua --scale 4 --precision int8 --color-fix none
python -m winapp.cli --video clip.mp4 --dry-run --json      # xem trước, không đẩy kernel
```

| File | Việc |
|---|---|
| `run_win.py`, `winapp/app.py` | giao diện Tkinter (thuần thư viện chuẩn Python) |
| `winapp/job.py` | ghép job: share → tunnel → push → đợi → tải kết quả → dọn dẹp |
| `winapp/kaggle_client.py` | bọc CLI `kaggle`; dựng thư mục kernel + chèn tham số vào worker |
| `winapp/share.py`, `winapp/tunnel.py` | HTTP server 1 file (có token) + cloudflared quick tunnel |
| `winapp/cli.py` | chạy không GUI: `--check`, `--tunnel-test`, `--dry-run`, chạy thật |
| `Chay_App.cmd`, `Chay_App.pyw` | nhấp đúp để mở app |
| `build_exe.cmd` | build `dist\ValoUpscale.exe` (1 file, nhấp đúp) |
| `kaggle_jobs/recavsr_worker.py` | script chạy **trên Kaggle**: ComfyUI + node pack + weight + upscale |
| `valo_upscale/recavsr.py` | module dùng chung: tải weight int8/bf16, dựng workflow, gọi ComfyUI |
| `recavsr_kaggle.ipynb` | chạy tay trên Kaggle (cùng worker, không cần app) |

### Tham số (mặc định đã tốt cho T4)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `--scale` | `4` | model huấn luyện ở 4×; T4 16 GB nên dùng `2` nếu input ≥ 640×360 |
| `--precision` | `int8` | 6,27 GB; `bf16` là bản gốc 10,6 GB |
| `--kernels` | `fast` | `exact` = đúng tham chiếu, chậm hơn |
| `--decoder` | `auto` | `auto` = FlashDecoder, tự chuyển `vae` khi GPU không chạy được FlashDecoder (T4 ở 4×); `vae` chậm hơn 25–40× nhưng nhiều chi tiết hơn |
| `--color-fix` | `none` | `adain` / `wavelet` để khớp màu với video gốc |
| `--encoder` | `auto` | `auto` → HEVC NVENC → H.264 NVENC → x264; chọn `h264` nếu cần tương thích tối đa |
| `--kv-offload` | tắt | bật khi output không đủ VRAM (kết quả y hệt) |
| `--no-lowvram` | (đang bật lowvram) | tắt nếu VRAM thoải mái |

VRAM tham chiếu của tác giả (int8 + FlashDecoder, fast kernels): **~11,6 GB ở 1440×1440** (vừa T4),
**~21 GB ở 3840×1920** (không vừa T4 → node tự hạ cấu hình hoặc báo lỗi nêu rõ scale lớn nhất chạy được).

App có **ước lượng VRAM trước khi chạy** (`estimate_recavsr_vram_gb`, khớp đúng 2 điểm tham chiếu của tác giả)
và in cảnh báo ngay trong kế hoạch dry-run, ví dụ: `1920x1080 scale 4 int8 → 3840x2160 ~22,2 GB > 16 GB
→ Nên dùng --scale 1 (~11,6 GB)`. Tránh việc đợi ~4 phút cài đặt trên Kaggle rồi mới biết là hết VRAM.

### ⚠️ FlashDecoder trên T4 (sm_75) — dùng `--decoder vae` cho 4×

Node pack ghi rõ: *"the decoder's window kernel needs compute capability 8.0 (RTX 30 series or newer)"*.
T4 là **sm_75** nên kernel cửa sổ của FlashDecoder vượt **shared memory** 64 KB của card khi ảnh ra lớn:

```
ComfyUI lỗi ở node ReCaVSRUpscale: out of resource: shared memory, Required: 98304,
Hardware limit: 65536. (triton.runtime.errors.OutOfResources)
```

Số đo thật: **2× thì FlashDecoder vừa** (chạy tốt), **4× thì vượt** → chết ở node `ReCaVSRUpscale`.

**Bằng chứng từ các lần chạy thật**: ở 4× thì lỗi này **không phải của FlashDecoder** — đổi sang `decoder=vae`
(app đã tự tải VAE 1,34 GB và chạy lại) **vẫn cùng lỗi y hệt**, vì kernel Triton đó nằm ở đường **attention của
`kernels=fast`** mà cả hai decoder đều dùng (`Required: 98304 …` giống nhau). Đòn đúng là **`--kernels exact`**
(chuỗi op của bản tham chiếu, không Triton) — và **đã kiểm chứng**: 4× chỉ chạy được khi qua `exact` (265 s).

Vì vậy app tự đi **thang thử** khi gặp đúng lỗi shared memory (chỉ khi bạn để `--decoder auto`):

| Bước | decoder | kernels | Ghi chú |
|---|---|---|---|
| 1 | flashdecoder | `fast` (như bạn chọn) | nhanh nhất |
| 2 | flashdecoder | `exact` | khi bước 1 gặp lỗi shared memory — không cần tải gì thêm |
| 3 | vae | `exact` | tự tải `wan2.2_vae.safetensors` (1,34 GB) của Comfy-Org vào `models/vae`; chậm hơn 25–40× phần decode nhưng nhiều chi tiết hơn |

Lỗi khác (không phải shared memory) thì **không thử lại** — ném nguyên lỗi để bạn thấy đúng nguyên nhân.
Nếu bạn chỉ định rõ `--decoder flashdecoder` thì thang chỉ có bước 1–2 (không tự đổi sang VAE).

Chỉ định tay: `--kernels fast | exact` và `--decoder flashdecoder | vae | auto`.

### Đã kiểm chứng (máy dev, 62 checks)

| Hạng mục | Kết quả |
|---|---|
| Node pack thật trong ComfyUI 0.44 | ComfyUI nạp được `ReCaVSRLoader` + `ReCaVSRUpscale`, input khớp graph; workflow dựng bằng code **trùng file API chính thức** và **VALID** (lỗi chỉ ở khâu nạp weight giả) |
| Đọc combo của schema V3 | sửa `_combo_options` để đọc `["COMBO", {"options": […]}]` — nếu không, worker sẽ báo thiếu weight dù có đủ |
| Weight | đúng repo/đúng tên file (`PedroMarinhoDev/ReCaVSR-ComfyUI` int8 + `kopper/ReCaVSR` prompt/flashdecoder), **3/3 file có thật trên Hub**, có token, dùng lại bản sẵn có (symlink) |
| Ứng dụng Windows (30 checks: thêm Decoder GUI, tự hồi phục tunnel 7b, ước lượng VRAM trong dry-run) | chèn tham số vào worker rồi **compile được**; metadata kernel đúng (GPU/internet/script); client Kaggle: push → status (`queued→running→complete`) → output → copy kết quả; HTTP server 1 file (200/401/HEAD/health); parser URL tunnel; `Runner.run()` chạy trọn đường ống với CLI kaggle **giả**; dry-run **không** đẩy kernel và **vẫn dọn** tunnel |
| Tunnel thật | tải `cloudflared.exe` (53 MB), mở quick tunnel, `/health` OK qua URL công khai và **tải được file qua URL đó** (`payload_ok: true`) |
| Tốc độ đường truyền video | đo thật với file 50 MB qua URL công khai của tunnel: **12,1 MB/s** (4,1 s) → video 200 MB ≈ **20 s**; video lớn không phải là điểm nghẽn |
| Kaggle thật — đăng nhập | `kaggle.json` (username + key) → `kaggle config view` + `kaggle kernels list -m` trả về kernel thật của tài khoản |
| Kaggle thật — push→chạy→tải kết quả | kernel CPU `hoanghoannguyen1223/valo-smoke`: push OK, trạng thái `RUNNING → COMPLETE`, tải về `valo_result.json` + file sinh ra trên Kaggle (`hello from kaggle …`) |
| Kaggle thật — job ReCaVSR int8 **2×** (GPU T4) | **chạy trọn đường ống, có kết quả**: clip 320×180/24 frame → **640×360, 24 frame, 12 fps, còn nguyên audio**; suy luận 36,8 s, cả job 163,8 s (kể cả tải 6,27 GB weight + cài ComfyUI). App tải về `sample_clip_recavsr_x2.mp4` (281 KB) |
| Kaggle thật — job ReCaVSR int8 **4×** (GPU T4) | **chạy được nhờ thang thử**: `kernels=fast` hỏng ở node `ReCaVSRUpscale` (`out of resource: shared memory, Required: 98304, Hardware limit: 65536`), app **tự chuyển `kernels=exact`** rồi chạy xong → **1280×720, 24 frame, 12 fps, còn nguyên audio**; suy luận **265,3 s** (≈11 s/frame), cả job 409,2 s. Log ghi rõ: `fallback: Triton hết shared memory ở bước này (decoder flashdecoder, kernels fast) → thử tiếp: … kernels exact` |
| Audio sau upscale | giữ nguyên nội dung (**tương quan phổ 0,9997** với bản gốc), mức âm lượng không đổi (−21,1 → −21,2 dB), chỉ dài thêm 23 ms |
| Đo độ lệch A/V của bộ ghi node pack | tách riêng ra đo bằng clip có **transient** (`samples/sync_click.mp4`: ô trắng nhấp nháy + bíp 1 kHz cùng ở 1,000 s): bộ ghi của node pack lệch **+23 ms** — đúng bằng độ trễ mồi của AAC (1024 mẫu / 44,1 kHz) nên không đáng kể. ⚠️ Trước đó tôi đo nhầm −230 ms vì dùng **tương quan trên sine 440 Hz** (tuần hoàn → đỉnh không xác định) — số đó **không** đúng |
| Chất lượng so với Lanczos | lệch trung bình 8,86/255 và **bị biến dạng chữ timecode + nhoè ô vuông nhỏ** trên pattern tổng hợp — đúng hành vi của model phục hồi sinh tạo với nội dung 1-pixel (không phải ảnh thật), không phải lỗi pipeline. Màu/khung hình vẫn đúng |
| Năm lỗi chỉ lộ khi chạy thật (đã sửa) | (1) worker **bỏ qua `--video-url`** vì argparse không lấy default từ `JOB` → app chia sẻ video qua tunnel nhưng kernel không tải; (2) weight int8 bị lưu **lồng thư mục** `models/recavsr/recavsr/…` (`hf_hub_download` ghép `<local_dir>/<remote>`) → ComfyUI không thấy; (3) worker để lại **1544 file / ~7 GB** (ComfyUI + weight) trong `/kaggle/working` → `kaggle kernels output` tải cả cây; (4) `recavsr.upscale_video` truyền tên transformer qua `model_file` → client chung kiểm nó trong combo **của RelayVSR** (`models/diffusion_models`) và báo thiếu weight dù đã có → thêm `extra_combos` + `allow_default_weights=False`; (5) `upscale_video(progress=…)` là callback **3 tham số** nhưng `wait_prompt` gọi **2 tham số** → `TypeError` ngay khi chờ ComfyUI (đã bọc lại) |
| Cách tải output đã gọn lại | worker **dọn ComfyUI/weight** trong `finally` (cờ `keep_comfy`, mặc định tắt) → output chỉ còn video + `valo_result.json` + `valo_run.log`; app còn có `KaggleCli.fetch_files()` dùng `kagglesdk` để **chỉ tải file json/log/mp4** |
| Khi job lỗi, app hiện lý do | `Runner.run()` tải `valo_result.json` + 12 dòng cuối `valo_run.log` ngay trong nhật ký app (trước đây phải mở Kaggle) |
| Suy luận ReCaVSR 2× trên T4 | **đã kiểm chứng thật** (xem bảng trên): 640×360, audio còn nguyên, 36,8 s cho 24 frame (≈1,5 s/frame ở 320×180, int8) — tốc độ thật để bạn ước lượng: video 10 s / 30 fps (300 frame) ≈ 7–8 phút suy luận |

Ghi chú về token Kaggle (đã thử thật): token kiểu mới `KGAT_…` **không** dùng được với CLI `kaggle` 1.7.4.5
(API trả 401 khi gọi kiểu Basic auth, 200 khi gửi `Authorization: Bearer <token>`; nhưng CLI bản này chỉ đọc
`kaggle.json`/`KAGGLE_USERNAME`+`KAGGLE_KEY`). Vì vậy app dùng `kaggle.json` — tải ở Kaggle → *Account → API →
Create New API Token* — rồi để vào `%USERPROFILE%\.kaggle\kaggle.json`. Nếu sau này Kaggle chỉ cấp token `KGAT_…`,
đường Bearer có sẵn trong `kagglesdk` (biến môi trường `KAGGLE_API_TOKEN`) và có thể nối thêm vào `winapp`.

Hai chi tiết chỉ lộ ra khi chạy thật trên Kaggle:
**Kaggle slugify `title` để đặt slug kernel** (title khác slug → kernel bị đổi tên, và `kernels push` in ra ref
mới) → app đặt `title` suy ra từ slug và luôn dùng **ref do push trả về**; và worker nhận tham số qua literal
Python chứ không phải JSON (JSON sinh `false/true` → `NameError` khi chạy).

Ngoài ra khi viết code đã bắt được 2 lỗi thật: node pack **không import được nếu thiếu `triton`**
(`flashdecoder.py` import `tl`) → worker cài triton **bắt buộc**; và HTTP reason phrase tiếng Việt làm
`send_error` ném `UnicodeEncodeError` (HTTP chỉ nhận latin-1) → đã đổi sang tiếng Anh.
