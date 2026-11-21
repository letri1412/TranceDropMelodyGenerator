"""Output watermark: every MIDI we write carries the tool/author/canary."""

from __future__ import annotations

import os

AUTHOR = "letri1412"
TOOL = "TranceDropMelodyGenerator"
DEFAULT_VERSION = "v2"
CANARY = "TRANCE-LORA-V2-CANARY-20261122-7f3a9c"
REPO = "https://github.com/letri1412/TranceDropMelodyGenerator"


def watermark_text(version: str | None = None) -> str:
    v = version or os.environ.get("TRANCE_WATERMARK_VERSION") or DEFAULT_VERSION
    return f"{TOOL} {v} | author: {AUTHOR} | {CANARY} | {REPO}"


def copyright_text() -> str:
    return f"(c) 2026 {AUTHOR} - {TOOL} (code Apache-2.0, weights OpenRAIL-M)"
