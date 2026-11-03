"""Đọc media từ thư mục công khai + đẩy kết quả ngược lên nơi lưu trữ.

Ràng buộc thực tế (quan trọng):
- Link công khai CHỈ cho phép ĐỌC. Muốn upload ngược lại đúng thư mục đó thì phải có
  quyền ghi: Service Account của Google Drive, Dropbox access token, hoặc GitHub PAT.
- Google Drive : đọc thư mục "Anyone with the link" bằng `gdown`; ghi bằng Service Account.
- Dropbox     : đọc/ghi bằng access token (thư mục phải nằm trong tài khoản của bạn,
                hoặc bạn đã add thư mục được chia sẻ vào Dropbox).
- GitHub      : đọc repo public không cần token; ghi cần PAT (`Contents: Read and write`).
- HTTP        : đọc trang index thư mục / danh sách URL / URL file trực tiếp; ghi không hỗ trợ.
- kaggle_local: kết quả nằm trong /kaggle/working (tải ở tab Output sau khi Save version).
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

from . import probe

UA = {"User-Agent": "valo-upscale/1.0"}
MEDIA_EXT = set(probe.VIDEO_EXT) | set(probe.IMAGE_EXT)
GITHUB_MAX_FILE_MB = 100
DROPBOX_MAX_FILE_MB = 150


def github_bases(api_base=None, raw_base=None):
    """Endpoint GitHub — đọc env lúc gọi để đổi được sang proxy/GH Enterprise."""
    return (
        (api_base or os.environ.get("GITHUB_API_BASE", "https://api.github.com")).rstrip("/"),
        (raw_base or os.environ.get("GITHUB_RAW_BASE", "https://raw.githubusercontent.com")).rstrip("/"),
    )


def dropbox_bases(api_base=None, content_base=None):
    return (
        (api_base or os.environ.get("DROPBOX_API_BASE", "https://api.dropboxapi.com/2")).rstrip("/"),
        (content_base or os.environ.get("DROPBOX_CONTENT_BASE", "https://content.dropboxapi.com/2")).rstrip("/"),
    )


class RemoteError(RuntimeError):
    """Lỗi khi đọc/ghi với nơi lưu trữ (thông báo đã rõ ràng cho người dùng)."""


@dataclass
class RemoteItem:
    """Một file media tìm thấy ở nguồn từ xa."""

    name: str
    provider: str = "http"
    size: int | None = None
    download_url: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return probe.media_kind(self.name)

    @property
    def size_text(self) -> str:
        if not self.size:
            return "?"
        return f"{self.size / 1048576:.1f} MB" if self.size >= 1048576 else f"{self.size / 1024:.0f} KB"


def is_media(name: str) -> bool:
    return Path(name).suffix.lower() in MEDIA_EXT


def detect_provider(source: str) -> str:
    url = source.strip()
    if "drive.google.com" in url or "docs.google.com" in url:
        return "gdrive"
    if "github.com" in url:
        return "github"
    if "dropbox.com" in url:
        return "dropbox"
    return "http"


# --------------------------------------------------------------------------- GitHub

_GITHUB_RE = re.compile(
    r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+?)(?:\.git)?"
    r"(?:/(?P<kind>tree|blob)/(?P<ref>[^/]+)(?:/(?P<path>[^?#]*))?)?(?:[?#]|$)"
)


def _github_headers(token=None):
    headers = dict(UA, Accept="application/vnd.github+json")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def list_github(source, token=None, api_base=None, raw_base=None):
    api_base, raw_base = github_bases(api_base, raw_base)
    match = _GITHUB_RE.search(source.strip())
    if not match:
        raise RemoteError(f"Link GitHub không hợp lệ: {source}")

    owner, repo = match.group("owner"), match.group("repo")
    kind, ref, sub = match.group("kind"), match.group("ref"), (match.group("path") or "").strip("/")

    if not ref:
        info = requests.get(f"{api_base}/repos/{owner}/{repo}", headers=_github_headers(token), timeout=30)
        if info.status_code == 404:
            raise RemoteError(f"Không thấy repo {owner}/{repo} (repo private thì cần token).")
        info.raise_for_status()
        ref = info.json().get("default_branch", "main")

    if kind == "blob":
        name = Path(sub).name
        raw = f"{raw_base}/{owner}/{repo}/{ref}/{sub}"
        return [RemoteItem(name=name, provider="github", download_url=raw, extra={"path": sub})]

    tree = requests.get(
        f"{api_base}/repos/{owner}/{repo}/git/trees/{ref}",
        params={"recursive": "1"},
        headers=_github_headers(token),
        timeout=60,
    )
    if tree.status_code == 404:
        raise RemoteError(f"Không đọc được cây file của {owner}/{repo}@{ref}.")
    tree.raise_for_status()

    items = []
    for node in tree.json().get("tree", []):
        path = node.get("path", "")
        if node.get("type") != "blob" or not is_media(path):
            continue
        if sub and not path.startswith(sub + "/") and path != sub:
            continue
        items.append(
            RemoteItem(
                name=Path(path).name,
                provider="github",
                size=node.get("size"),
                download_url=f"{raw_base}/{owner}/{repo}/{ref}/{path}",
                extra={"path": path, "owner": owner, "repo": repo, "ref": ref},
            )
        )
    return sorted(items, key=lambda item: item.name)


# --------------------------------------------------------------------------- Dropbox


def _dropbox_headers(token):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def list_dropbox(source, token, api_base=None, content_base=None):
    if not token:
        raise RemoteError(
            "Dropbox cần access token (dropbox.com/developers/apps → Create app → Generated access token, "
            "scope files.content.read + files.content.write)."
        )
    api_base, _ = dropbox_bases(api_base=api_base)

    meta = requests.post(
        f"{api_base}/sharing/get_shared_link_metadata",
        headers=_dropbox_headers(token),
        data=json.dumps({"url": source.strip()}),
        timeout=45,
    )
    if meta.status_code != 200:
        raise RemoteError(
            "Không đọc được link Dropbox này bằng token của bạn "
            f"({meta.status_code}: {meta.text[:200]}). Hãy 'Add to your Dropbox' thư mục được chia sẻ trước."
        )
    info = meta.json()
    if info.get(".tag") == "file":
        path = info.get("path_lower") or info.get("path_display")
        return [RemoteItem(name=Path(path).name, provider="dropbox", size=info.get("size"),
                           extra={"dropbox_path": path})]

    path = info.get("path_lower") or ""
    entries, cursor = [], None
    while True:
        if cursor is None:
            res = requests.post(
                f"{api_base}/files/list_folder",
                headers=_dropbox_headers(token),
                data=json.dumps({"path": path, "recursive": True, "limit": 2000}),
                timeout=60,
            )
        else:
            res = requests.post(
                f"{api_base}/files/list_continue",
                headers=_dropbox_headers(token),
                data=json.dumps({"cursor": cursor}),
                timeout=60,
            )
        if res.status_code != 200:
            raise RemoteError(f"Dropbox list_folder lỗi {res.status_code}: {res.text[:200]}")
        payload = res.json()
        for entry in payload.get("entries", []):
            if entry.get(".tag") == "file" and is_media(entry.get("name", "")):
                entries.append(
                    RemoteItem(
                        name=entry["name"],
                        provider="dropbox",
                        size=entry.get("size"),
                        extra={"dropbox_path": entry.get("path_lower") or entry.get("path_display")},
                    )
                )
        if not payload.get("has_more"):
            break
        cursor = payload.get("cursor")
    return sorted(entries, key=lambda item: item.name)


def _dropbox_download(item, dest, token, content_base=None):
    _, content_base = dropbox_bases(content_base=content_base)
    with requests.post(
        f"{content_base}/files/download",
        headers={"Authorization": f"Bearer {token}", "Dropbox-API-Arg": json.dumps({"path": item.extra["dropbox_path"]})},
        stream=True,
        timeout=300,
    ) as res:
        if res.status_code != 200:
            raise RemoteError(f"Dropbox download lỗi {res.status_code}: {res.text[:200]}")
        with open(dest, "wb") as handle:
            for chunk in res.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    return dest


# --------------------------------------------------------------------------- HTTP


def _urls_from_text(text):
    """Đọc danh sách URL trong file .txt (mỗi dòng một URL)."""
    items = []
    for line in text.splitlines():
        candidate = line.strip()
        if candidate.startswith(("http://", "https://")) and is_media(candidate):
            items.append(
                RemoteItem(name=unquote(Path(urlparse(candidate).path).name), provider="http",
                           download_url=candidate)
            )
    return items


def list_http(source, session=None):
    session = session or requests
    url = source.strip()
    if "\n" in url or "," in url:  # danh sách URL dán trực tiếp
        return [item for piece in re.split(r"[\s,]+", url) if piece.strip()
                for item in list_http(piece.strip(), session=session)]

    with session.get(url, headers=UA, timeout=60, stream=True) as res:
        if res.status_code != 200:
            raise RemoteError(f"Không tải được {url} (HTTP {res.status_code})")
        ctype = (res.headers.get("content-type") or "").lower()
        name = unquote(Path(urlparse(url).path).name)

        if is_media(name) or ctype.startswith(("image/", "video/", "application/octet-stream")):
            return [RemoteItem(name=name or "download", provider="http", download_url=url)]

        text = res.text or ""

    if "text/plain" in ctype:
        return _urls_from_text(text)

    items, seen = [], set()
    for href in re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.IGNORECASE):
        if href.startswith(("?", "#", "mailto:", "javascript:")):
            continue
        raw = urljoin(url if url.endswith("/") else url + "/", href).split("#")[0]
        candidate = unquote(Path(urlparse(raw).path).name)
        if not is_media(candidate) or raw in seen:
            continue
        seen.add(raw)
        items.append(RemoteItem(name=candidate, provider="http", download_url=raw))
    return sorted(items, key=lambda item: item.name)


# --------------------------------------------------------------------------- Google Drive


def list_gdrive(source, gdown_module=None):
    """Liệt kê thư mục Drive công khai bằng gdown (skip_download=True)."""
    if gdown_module is None:
        try:
            import gdown  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RemoteError("Thiếu gdown. Cài: pip install -q gdown") from exc
        gdown_module = gdown

    url = source.strip()
    file_match = re.search(r"/(?:file|document)/d/([\w-]{10,})", url)
    if file_match:
        return [RemoteItem(name="", provider="gdrive", extra={"file_id": file_match.group(1)})]

    try:
        found = gdown_module.download_folder(url=url, quiet=True, skip_download=True)
    except TypeError:  # gdown cũ không có skip_download
        raise RemoteError("gdown cũ không hỗ trợ xem trước. Cài lại: pip install -U gdown") from None
    except Exception as exc:
        raise RemoteError(
            f"Không đọc được thư mục Drive: {exc}. Kiểm tra quyền 'Anyone with the link' của thư mục."
        ) from exc

    items = []
    for entry in found or []:
        name = getattr(entry, "path", "") or ""
        if not is_media(name):
            continue
        items.append(RemoteItem(name=Path(name).name, provider="gdrive", extra={"file_id": entry.id}))
    return sorted(items, key=lambda item: item.name)


def _download_gdrive(item, dest, gdown_module=None):
    if gdown_module is None:
        import gdown  # noqa: PLC0415

        gdown_module = gdown
    url = f"https://drive.google.com/uc?id={item.extra['file_id']}"
    output = str(dest) + ("" if Path(dest).suffix else ".bin")
    result = gdown_module.download(url=url, output=output, quiet=True)
    if not result:
        raise RemoteError(f"gdown không tải được file id {item.extra['file_id']}")
    return Path(result)


# --------------------------------------------------------------------------- facade


def list_remote(source, *, dropbox_token=None, github_token=None, api_base=None, raw_base=None,
                content_base=None, gdown_module=None, session=None):
    """Liệt kê media trong thư mục/link công khai (tự nhận provider)."""
    source = (source or "").strip()
    if not source:
        raise RemoteError("Chưa nhập link thư mục.")

    provider = detect_provider(source)
    if provider == "gdrive":
        return list_gdrive(source, gdown_module=gdown_module)
    if provider == "github":
        return list_github(source, token=github_token, api_base=api_base, raw_base=raw_base)
    if provider == "dropbox":
        return list_dropbox(source, dropbox_token, api_base=api_base)
    return list_http(source, session=session)


def download_remote(items, dest_dir, *, dropbox_token=None, gdown_module=None, progress=None, session=None,
                    content_base=None):
    """Tải các RemoteItem vào dest_dir, trả list đường dẫn local."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests
    paths = []

    for index, item in enumerate(items, start=1):
        name = item.name or f"file_{index}"
        target = dest_dir / name
        if target.exists() and target.stat().st_size > 0:
            paths.append(target)
            continue

        if item.provider == "gdrive":
            paths.append(Path(_download_gdrive(item, target, gdown_module=gdown_module)))
        elif item.provider == "dropbox":
            paths.append(Path(_dropbox_download(item, target, dropbox_token, content_base=content_base)))
        else:
            with session.get(item.download_url, headers=UA, stream=True, timeout=300) as res:
                if res.status_code != 200:
                    raise RemoteError(f"Tải {item.name} lỗi HTTP {res.status_code}")
                with open(target, "wb") as handle:
                    for chunk in res.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            paths.append(target)

        if progress:
            progress(f"  ↓ {index}/{len(items)} {name} ({item.size_text})")
    return paths


# --------------------------------------------------------------------------- upload ngược


def upload_kaggle_local(paths, dest_dir):
    """Giữ kết quả trong /kaggle/working (hoặc thư mục chỉ định) — luôn dùng được, không cần token."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for path in paths:
        path = Path(path)
        target = dest_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        out.append(str(target))
    return out


_DRIVE_ID_RE = re.compile(r"[\w-]{10,}")


def parse_drive_folder_id(folder):
    """Nhận link thư mục, link có ?id=, hoặc ID thuần. Trả None nếu rỗng."""
    text = (folder or "").strip()
    if not text:
        return None
    match = re.search(r"/folders/([\w-]{10,})", text) or re.search(r"[?&]id=([\w-]{10,})", text)
    if match:
        return match.group(1)
    if _DRIVE_ID_RE.fullmatch(text):
        return text
    raise RemoteError(
        f"Không nhận ra ID thư mục Google Drive từ '{folder}'. "
        "Dán link dạng https://drive.google.com/drive/folders/<ID> hoặc chính <ID> (dài ~33 ký tự)."
    )


def load_service_account(sa_json):
    """Nhận nội dung JSON, đường dẫn file, hoặc biến môi trường GDRIVE_SA_JSON."""
    sa_json = sa_json or os.environ.get("GDRIVE_SA_JSON", "")
    if not sa_json:
        raise RemoteError(
            "Thiếu Service Account JSON. Tạo tại console.cloud.google.com → IAM → Service Accounts → Keys "
            "(JSON), bật Drive API, rồi share thư mục Drive cho email service account (quyền Editor)."
        )
    text = sa_json.strip()
    if text.startswith("{"):
        return json.loads(text)
    path = Path(text)
    if not path.exists():
        raise RemoteError(f"Không thấy file Service Account JSON: {text}")
    return json.loads(path.read_text(encoding="utf-8"))


def upload_gdrive(paths, folder, sa_json=None, service_factory=None, progress=None):
    """Upload kết quả vào thư mục Drive (cần Service Account)."""
    info = load_service_account(sa_json)
    folder_id = parse_drive_folder_id(folder)

    if service_factory is None:
        try:
            from google.oauth2 import service_account  # noqa: PLC0415
            from googleapiclient.discovery import build  # noqa: PLC0415
            from googleapiclient.http import MediaFileUpload  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RemoteError("Thiếu thư viện Google. Cài: pip install -q google-api-python-client google-auth") from exc

        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
    else:
        service = service_factory(info)
        from googleapiclient.http import MediaFileUpload  # noqa: PLC0415

    created = []
    for path in paths:
        media = MediaFileUpload(str(path), mimetype=mimetypes.guess_type(str(path))[0] or "application/octet-stream",
                                resumable=True)
        body = {"name": Path(path).name}
        if folder_id:
            body["parents"] = [folder_id]
        result = service.files().create(body=body, media_body=media, fields="id,webViewLink").execute()
        created.append(result.get("webViewLink") or result.get("id"))
        if progress:
            progress(f"  ↑ Drive: {Path(path).name}")
    return created


def upload_dropbox(paths, folder, token, api_base=None, content_base=None, progress=None):
    """Upload kết quả vào thư mục Dropbox (path trong tài khoản của bạn)."""
    if not token:
        raise RemoteError("Thiếu Dropbox access token.")
    api_base, _ = dropbox_bases(api_base=api_base)
    _, content_base = dropbox_bases(content_base=content_base)
    base_path = "/" + (folder or "").strip().strip("/")

    created = []
    for path in paths:
        path = Path(path)
        size_mb = path.stat().st_size / 1048576
        if size_mb > DROPBOX_MAX_FILE_MB:
            raise RemoteError(
                f"{path.name} nặng {size_mb:.0f} MB, vượt giới hạn upload đơn 150 MB của Dropbox. "
                "Hãy dùng scale nhỏ hơn hoặc tải về /kaggle/working."
            )
        target = f"{base_path}/{path.name}"
        with open(path, "rb") as handle:
            res = requests.post(
                f"{content_base}/files/upload",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Dropbox-API-Arg": json.dumps({"path": target, "mode": "overwrite", "autorename": True}),
                    "Content-Type": "application/octet-stream",
                },
                data=handle,
                timeout=600,
            )
        if res.status_code != 200:
            raise RemoteError(f"Dropbox upload lỗi {res.status_code}: {res.text[:200]}")
        created.append(res.json().get("path_display") or target)
        if progress:
            progress(f"  ↑ Dropbox: {target}")
    return created


def upload_github(paths, folder, token, api_base=None, commit_message=None, progress=None):
    """Upload kết quả vào repo GitHub (folder dạng 'owner/repo/duong/dan'), cần PAT."""
    if not token:
        raise RemoteError("Thiếu GitHub token (PAT có quyền Contents: Read and write).")
    api_base, _ = github_bases(api_base=api_base)
    parts = [p for p in (folder or "").split("/") if p]
    if len(parts) < 2:
        raise RemoteError("Thư mục GitHub phải có dạng owner/repo[/duong/dan].")
    owner, repo, sub = parts[0], parts[1], "/".join(parts[2:])
    message = commit_message or f"Upscale {time.strftime('%Y-%m-%d %H:%M')}"

    created = []
    for path in paths:
        path = Path(path)
        size_mb = path.stat().st_size / 1048576
        if size_mb > GITHUB_MAX_FILE_MB:
            raise RemoteError(f"{path.name} nặng {size_mb:.0f} MB, vượt giới hạn 100 MB/file của GitHub.")
        target = f"{sub}/{path.name}" if sub else path.name
        api_url = f"{api_base}/repos/{owner}/{repo}/contents/{target}"
        body = {
            "message": message,
            "content": base64.b64encode(path.read_bytes()).decode(),
        }
        existing = requests.get(api_url, headers=_github_headers(token), timeout=60)
        if existing.status_code == 200:
            body["sha"] = existing.json()["sha"]
        res = requests.put(api_url, headers=_github_headers(token), json=body, timeout=600)
        if res.status_code not in (200, 201):
            raise RemoteError(f"GitHub upload lỗi {res.status_code}: {res.text[:200]}")
        created.append(res.json().get("content", {}).get("html_url") or target)
        if progress:
            progress(f"  ↑ GitHub: {target}")
    return created


def upload_results(paths, *, kind="kaggle_local", folder=None, dropbox_token=None, github_token=None,
                   drive_sa_json=None, api_base=None, content_base=None, service_factory=None, progress=None):
    """Đẩy kết quả ngược lên nơi lưu trữ theo `kind`."""
    if not paths:
        return []
    if kind == "kaggle_local":
        return upload_kaggle_local(paths, folder or "/kaggle/working/upscaled")
    if kind == "gdrive":
        return upload_gdrive(paths, folder, sa_json=drive_sa_json, service_factory=service_factory, progress=progress)
    if kind == "dropbox":
        return upload_dropbox(paths, folder, dropbox_token, api_base=api_base, content_base=content_base,
                              progress=progress)
    if kind == "github":
        return upload_github(paths, folder, github_token, api_base=api_base, progress=progress)
    raise RemoteError(f"Kiểu đích không hỗ trợ: {kind}")
