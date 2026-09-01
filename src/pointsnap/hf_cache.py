"""Redirects the Hugging Face cache to this project's own directory on D:,
instead of the account-wide default (~/.cache/huggingface, on a system
drive that runs with very little headroom on this machine). Must be called
before `transformers`/`huggingface_hub` is imported anywhere in the
process, since their cache-path constants are resolved at import time --
every backbone wrapper calls this as the first thing it does, before its
own `transformers` import.
"""

from __future__ import annotations

import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parents[2] / ".hf_cache"


def configure_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", str(_CACHE_DIR))
