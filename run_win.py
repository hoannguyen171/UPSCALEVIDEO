"""Mở ứng dụng: upscale video bằng GPU Kaggle (ReCaVSR int8).

    python run_win.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from winapp.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
