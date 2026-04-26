from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def get_hf_token() -> str | None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN")
    return token or None


def get_hf_kwargs() -> dict[str, str]:
    token = get_hf_token()
    return {"token": token} if token else {}


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path