"""Project-root-relative configuration paths shared by the API and dashboard."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(setting: str, default: str) -> Path:
    path = Path(os.getenv(setting, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
