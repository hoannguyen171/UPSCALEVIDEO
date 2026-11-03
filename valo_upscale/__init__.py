"""valo-upscale-gpu — bản port Python của app "Valo Upscale".

Upscale ảnh & video bằng Real-ESRGAN (PyTorch/CUDA) hoặc FFmpeg Lanczos+Unsharp,
chạy được trên Google Colab / Kaggle / máy local.
"""

from .dims import calculate_image_dimensions, calculate_video_dimensions
from .engine import (
    RealESRGANEngine,
    UpscaleCancelled,
    process_media,
    upscale_image_ffmpeg,
    upscale_video_ffmpeg,
)
from .models import MODEL_REGISTRY, list_models, load_model
from .probe import get_media_info, media_kind

__version__ = "1.0.0"

__all__ = [
    "MODEL_REGISTRY",
    "RealESRGANEngine",
    "UpscaleCancelled",
    "calculate_image_dimensions",
    "calculate_video_dimensions",
    "get_media_info",
    "list_models",
    "load_model",
    "media_kind",
    "process_media",
    "upscale_image_ffmpeg",
    "upscale_video_ffmpeg",
]
