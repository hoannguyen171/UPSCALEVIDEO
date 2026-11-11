"""Nhấp đúp file này để mở ứng dụng (không hiện cửa sổ đen console)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from winapp.app import main

if __name__ == "__main__":
    raise SystemExit(main())
