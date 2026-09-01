"""
brand.py — The Polycule's visual identity, in one place.

Values come from the vault's "Visual style guide" note. Every graphic template
reads from here so a brand change is one edit, not twelve.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

# Trans flag core — non-negotiable per the style guide.
TRANS_BLUE = "#55CDFC"
TRANS_PINK = "#F7A8B8"
WHITE = "#FFFFFF"

# Supporting palette.
WARM_DARK = "#1F1B2E"
SOFT_LAVENDER = "#F2EEFF"
TEXT_DARK = "#2D2B35"

PALETTE = {
    "trans_blue": TRANS_BLUE,
    "trans_pink": TRANS_PINK,
    "white": WHITE,
    "warm_dark": WARM_DARK,
    "soft_lavender": SOFT_LAVENDER,
    "text_dark": TEXT_DARK,
}

# Per-speaker accent, so name cards and pull-quotes stay visually consistent.
SPEAKER_ACCENT = {
    "Heather": TRANS_BLUE,
    "Sophie": TRANS_PINK,
}

PRIMARY_FONT = "Fredoka"
FALLBACK_FONTS = ("DejaVu Sans", "Liberation Sans", "Noto Sans")


@lru_cache(maxsize=None)
def font_path(family: str = PRIMARY_FONT, weight: str = "SemiBold") -> str:
    """
    Resolve a font family to a file on disk via fontconfig, falling back through
    FALLBACK_FONTS. Raises if nothing usable is installed, because a silently
    substituted font produces off-brand cards that look fine until upload.
    """
    for candidate in (family, *FALLBACK_FONTS):
        query = f"{candidate}:style={weight}" if candidate == family else candidate
        try:
            out = subprocess.run(
                ["fc-match", "-f", "%{file}\t%{family}", query],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
        if "\t" not in out:
            continue
        path, matched = out.split("\t", 1)
        # fc-match always returns *something*; only accept a real family match.
        if candidate.lower() in matched.lower() and Path(path).exists():
            return path
    raise RuntimeError(
        f"no usable font found for {family!r} or fallbacks {FALLBACK_FONTS}; "
        f"install Fredoka (see the Visual style guide note)"
    )


def has_brand_font() -> bool:
    """True when the real Fredoka is installed, rather than a fallback."""
    try:
        return PRIMARY_FONT.lower() in Path(font_path()).name.lower()
    except RuntimeError:
        return False


def rgba(hex_colour: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)
