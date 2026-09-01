"""
graphics.py — render per-episode overlay cards as transparent PNGs.

One renderer per overlay kind. Each takes the overlay's `fields` dict and
returns a path; the Kdenlive writer then drops the PNG on the graphics track
at the conformed timeline position. Cards are rendered at project resolution
so the editor never has to scale them.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .. import brand
from ..edl import Timeline

# Cards sit inside a safe margin so nothing clips on a phone screen.
SAFE_MARGIN = 0.06


def _font(size: int, weight: str = "SemiBold") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(brand.font_path(weight=weight), size)


def _wrap(draw, text, font, max_width) -> list[str]:
    """Greedy wrap by measured pixel width, not character count."""
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if draw.textlength(trial, font=font) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def _rounded_panel(draw, box, fill, radius=24, outline=None, width=0):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


# --- individual card renderers ---------------------------------------------

def render_name_card(img, fields, w, h):
    """Lower-third speaker label, accented in that speaker's brand colour."""
    name = fields.get("name", "")
    role = fields.get("role", "")
    accent = brand.SPEAKER_ACCENT.get(name, brand.TRANS_BLUE)
    d = ImageDraw.Draw(img)

    f_name = _font(int(h * 0.055))
    f_role = _font(int(h * 0.030), "Regular")
    x = int(w * SAFE_MARGIN)
    y = int(h * 0.74)

    tw = max(d.textlength(name, font=f_name), d.textlength(role, font=f_role))
    pad = int(h * 0.025)
    panel_h = int(h * 0.135) if role else int(h * 0.095)
    _rounded_panel(d, (x, y, x + tw + pad * 3, y + panel_h),
                   fill=brand.rgba(brand.WARM_DARK, 235), radius=int(h * 0.018))
    # Accent bar keys the card to the speaker.
    d.rounded_rectangle((x, y, x + int(w * 0.006), y + panel_h),
                        radius=int(h * 0.004), fill=brand.rgba(accent))

    d.text((x + pad * 2, y + pad * 0.8), name, font=f_name, fill=brand.rgba(brand.WHITE))
    if role:
        d.text((x + pad * 2, y + pad * 0.8 + f_name.size * 1.25), role,
               font=f_role, fill=brand.rgba(accent))


def render_citation(img, fields, w, h):
    """Source card: outlet, headline, date. Never rendered without human sign-off."""
    outlet = fields.get("outlet", "")
    headline = fields.get("headline", "")
    date = fields.get("date", "")
    d = ImageDraw.Draw(img)

    f_outlet = _font(int(h * 0.028))
    f_head = _font(int(h * 0.038), "Regular")
    box_w = int(w * 0.40)
    x = int(w * (1 - SAFE_MARGIN) - box_w)
    y = int(h * 0.10)
    pad = int(h * 0.028)

    lines = _wrap(d, headline, f_head, box_w - pad * 2)
    box_h = pad * 2 + f_outlet.size + int(f_head.size * 1.35) * len(lines) + f_outlet.size
    _rounded_panel(d, (x, y, x + box_w, y + box_h),
                   fill=brand.rgba(brand.SOFT_LAVENDER, 245), radius=int(h * 0.016))
    d.rounded_rectangle((x, y, x + box_w, y + int(h * 0.008)),
                        radius=int(h * 0.004), fill=brand.rgba(brand.TRANS_BLUE))

    ty = y + pad
    d.text((x + pad, ty), outlet.upper(), font=f_outlet, fill=brand.rgba(brand.TRANS_BLUE))
    ty += int(f_outlet.size * 1.6)
    for ln in lines:
        d.text((x + pad, ty), ln, font=f_head, fill=brand.rgba(brand.TEXT_DARK))
        ty += int(f_head.size * 1.35)
    if date:
        d.text((x + pad, ty), date, font=_font(int(h * 0.024), "Regular"),
               fill=brand.rgba(brand.TEXT_DARK, 170))


def render_pull_quote(img, fields, w, h):
    """Big centred opinion line, attributed."""
    quote = fields.get("quote", "")
    who = fields.get("speaker", "")
    accent = brand.SPEAKER_ACCENT.get(who, brand.TRANS_PINK)
    d = ImageDraw.Draw(img)

    f_q = _font(int(h * 0.075))
    max_w = int(w * (1 - SAFE_MARGIN * 2) * 0.85)
    lines = _wrap(d, f'"{quote}"', f_q, max_w)
    line_h = int(f_q.size * 1.25)
    total = line_h * len(lines)
    y = int(h * 0.62) - total // 2

    for ln in lines:
        tw = d.textlength(ln, font=f_q)
        x = (w - tw) / 2
        # Heavy shadow keeps white text legible over any footage.
        d.text((x + 4, y + 4), ln, font=f_q, fill=brand.rgba(brand.WARM_DARK, 190))
        d.text((x, y), ln, font=f_q, fill=brand.rgba(brand.WHITE))
        y += line_h
    if who:
        f_w = _font(int(h * 0.032), "Regular")
        tw = d.textlength(f"— {who}", font=f_w)
        d.text(((w - tw) / 2, y + int(h * 0.012)), f"— {who}",
               font=f_w, fill=brand.rgba(accent))


def render_chapter(img, fields, w, h):
    """Topic transition card — full-width band, reads at a glance."""
    title = fields.get("title", "")
    index = fields.get("index", "")
    d = ImageDraw.Draw(img)

    band_h = int(h * 0.22)
    y = int(h * 0.39)
    d.rectangle((0, y, w, y + band_h), fill=brand.rgba(brand.WARM_DARK, 240))
    d.rectangle((0, y, w, y + int(h * 0.010)), fill=brand.rgba(brand.TRANS_PINK))
    d.rectangle((0, y + band_h - int(h * 0.010), w, y + band_h),
                fill=brand.rgba(brand.TRANS_BLUE))

    f_t = _font(int(h * 0.070))
    lines = _wrap(d, title, f_t, int(w * 0.82))
    ty = y + (band_h - int(f_t.size * 1.2) * len(lines)) // 2
    if index:
        f_i = _font(int(h * 0.026), "Regular")
        d.text((int(w * SAFE_MARGIN), y + int(h * 0.028)), str(index).upper(),
               font=f_i, fill=brand.rgba(brand.TRANS_BLUE))
    for ln in lines:
        d.text(((w - d.textlength(ln, font=f_t)) / 2, ty), ln,
               font=f_t, fill=brand.rgba(brand.WHITE))
        ty += int(f_t.size * 1.2)


def render_stat(img, fields, w, h):
    """A number pulled out large, with its label and source underneath."""
    value = str(fields.get("value", ""))
    label = fields.get("label", "")
    source = fields.get("source", "")
    d = ImageDraw.Draw(img)

    f_v = _font(int(h * 0.20))
    f_l = _font(int(h * 0.040), "Regular")
    cx = int(w * 0.72)
    y = int(h * 0.28)

    vw = d.textlength(value, font=f_v)
    d.text((cx - vw / 2 + 5, y + 5), value, font=f_v, fill=brand.rgba(brand.WARM_DARK, 180))
    d.text((cx - vw / 2, y), value, font=f_v, fill=brand.rgba(brand.TRANS_BLUE))

    ly = y + int(f_v.size * 1.05)
    for ln in _wrap(d, label, f_l, int(w * 0.34)):
        d.text((cx - d.textlength(ln, font=f_l) / 2, ly), ln,
               font=f_l, fill=brand.rgba(brand.WHITE))
        ly += int(f_l.size * 1.3)
    if source:
        f_s = _font(int(h * 0.022), "Regular")
        d.text((cx - d.textlength(source, font=f_s) / 2, ly + int(h * 0.010)),
               source, font=f_s, fill=brand.rgba(brand.SOFT_LAVENDER, 190))


def render_reaction(img, fields, w, h):
    """Short reactive text, tilted, for a strong moment."""
    text = fields.get("text", "")
    f = _font(int(h * 0.090))
    scratch = Image.new("RGBA", (int(w * 0.6), int(h * 0.2)), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scratch)
    sd.text((10, 10), text, font=f, fill=brand.rgba(brand.WHITE),
            stroke_width=max(3, int(h * 0.006)), stroke_fill=brand.rgba(brand.WARM_DARK))
    scratch = scratch.rotate(-6, expand=True, resample=Image.BICUBIC)
    img.alpha_composite(scratch, (int(w * 0.30), int(h * 0.18)))


def render_poll(img, fields, w, h):
    """Comment-engagement prompt."""
    question = fields.get("question", "What do you think?")
    d = ImageDraw.Draw(img)
    f = _font(int(h * 0.042))
    lines = _wrap(d, question, f, int(w * 0.42))
    pad = int(h * 0.030)
    box_w = int(w * 0.48)
    box_h = pad * 2 + int(f.size * 1.3) * len(lines)
    x, y = int(w * SAFE_MARGIN), int(h * 0.16)
    _rounded_panel(d, (x, y, x + box_w, y + box_h),
                   fill=brand.rgba(brand.TRANS_PINK, 240), radius=int(h * 0.020))
    ty = y + pad
    for ln in lines:
        d.text((x + pad, ty), ln, font=f, fill=brand.rgba(brand.TEXT_DARK))
        ty += int(f.size * 1.3)


def render_further_reading(img, fields, w, h):
    """Points at a fuller article or a previous video."""
    title = fields.get("title", "")
    where = fields.get("where", "Link in the description")
    d = ImageDraw.Draw(img)
    f_t = _font(int(h * 0.034))
    f_w = _font(int(h * 0.024), "Regular")
    pad = int(h * 0.024)
    box_w = int(w * 0.42)
    lines = _wrap(d, title, f_t, box_w - pad * 2)
    box_h = pad * 2 + int(f_t.size * 1.3) * len(lines) + int(f_w.size * 1.4)
    x = int(w * (1 - SAFE_MARGIN)) - box_w
    y = int(h * 0.70)
    _rounded_panel(d, (x, y, x + box_w, y + box_h),
                   fill=brand.rgba(brand.WARM_DARK, 235), radius=int(h * 0.016),
                   outline=brand.rgba(brand.TRANS_BLUE), width=max(2, int(h * 0.003)))
    ty = y + pad
    for ln in lines:
        d.text((x + pad, ty), ln, font=f_t, fill=brand.rgba(brand.WHITE))
        ty += int(f_t.size * 1.3)
    d.text((x + pad, ty), where, font=f_w, fill=brand.rgba(brand.TRANS_BLUE))


def render_correction(img, fields, w, h):
    """Flags a claim as disputed or later corrected. Deliberately unmissable."""
    text = fields.get("text", "")
    d = ImageDraw.Draw(img)
    f_h = _font(int(h * 0.028))
    f_t = _font(int(h * 0.034), "Regular")
    pad = int(h * 0.026)
    box_w = int(w * 0.44)
    lines = _wrap(d, text, f_t, box_w - pad * 2)
    box_h = pad * 2 + int(f_h.size * 1.6) + int(f_t.size * 1.3) * len(lines)
    x = int((w - box_w) / 2)
    y = int(h * 0.06)
    _rounded_panel(d, (x, y, x + box_w, y + box_h),
                   fill=brand.rgba(brand.WHITE, 248), radius=int(h * 0.014),
                   outline=brand.rgba(brand.TRANS_PINK), width=max(3, int(h * 0.005)))
    d.text((x + pad, y + pad), fields.get("label", "CORRECTION"),
           font=f_h, fill=brand.rgba(brand.TRANS_PINK))
    ty = y + pad + int(f_h.size * 1.6)
    for ln in lines:
        d.text((x + pad, ty), ln, font=f_t, fill=brand.rgba(brand.TEXT_DARK))
        ty += int(f_t.size * 1.3)


def render_end_teaser(img, fields, w, h):
    """"Next week" teaser before the outro."""
    title = fields.get("title", "")
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, h), fill=brand.rgba(brand.WARM_DARK, 200))
    f_k = _font(int(h * 0.030), "Regular")
    f_t = _font(int(h * 0.080))
    kicker = fields.get("kicker", "NEXT WEEK")
    d.text(((w - d.textlength(kicker, font=f_k)) / 2, int(h * 0.36)), kicker,
           font=f_k, fill=brand.rgba(brand.TRANS_PINK))
    ty = int(h * 0.43)
    for ln in _wrap(d, title, f_t, int(w * 0.76)):
        d.text(((w - d.textlength(ln, font=f_t)) / 2, ty), ln,
               font=f_t, fill=brand.rgba(brand.WHITE))
        ty += int(f_t.size * 1.2)


RENDERERS = {
    "name_card": render_name_card,
    "citation": render_citation,
    "pull_quote": render_pull_quote,
    "chapter": render_chapter,
    "stat": render_stat,
    "reaction": render_reaction,
    "poll": render_poll,
    "further_reading": render_further_reading,
    "correction": render_correction,
    "end_teaser": render_end_teaser,
}


# --- stage entry point ------------------------------------------------------

def render_overlay(kind: str, fields: dict, out_dir: Path,
                   width: int, height: int) -> Path:
    """
    Render one card to a transparent PNG. The filename hashes the content, so
    re-running the stage after an unrelated edit reuses cards already made.
    """
    if kind not in RENDERERS:
        raise ValueError(f"no renderer for overlay kind {kind!r}")
    digest = hashlib.sha1(
        json.dumps({"k": kind, "f": fields, "w": width, "h": height}, sort_keys=True).encode()
    ).hexdigest()[:12]
    out = out_dir / f"{kind}_{digest}.png"
    if out.exists():
        return out

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    RENDERERS[kind](img, fields, width, height)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def run(timeline: Timeline, out_dir: Path) -> Timeline:
    """Render every overlay in the timeline and record the asset path on it."""
    out_dir = Path(out_dir)
    for ov in timeline.overlays:
        ov.asset = str(render_overlay(
            ov.kind, ov.fields, out_dir,
            timeline.episode.width, timeline.episode.height,
        ))
    return timeline
