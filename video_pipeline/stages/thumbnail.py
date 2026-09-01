"""
thumbnail.py — pick a frame, cut it out, and give it a die-cut sticker halo.

Two steps with a human in the middle, because the two halves fail differently.

`propose()` finds frames that are *technically* usable — eyes open, mouth not
mid-word, sharp, facing forward — and does it only inside windows the
transcript says are interesting, so the shortlist is drawn from moments that
carry the video rather than from the whole runtime. It writes a contact sheet
and stops.

`build()` takes the frame you chose and does the deterministic image work:
background removal, a hard alpha, the stroke stack, the shadow, the template.

Why the split: face scoring reliably rejects blinks and motion blur, but it
ranks *neutral* highest, and neutral is a bad thumbnail. So the score is used
as a filter for disqualifying faults, never as the final choice. You pick.

The halo
--------
The Legal Eagle look is a hard-edged sticker outline, not a glow. Two things
produce it, and both are easy to get wrong:

  1. The alpha must be *binary*. Background removal leaves a soft, partially
     transparent edge; dilating a soft matte spreads it like a blur and every
     corner and hair-tip rounds off into a blob. We threshold first.
  2. The growth must be a morphological dilate at constant width, not a
     Gaussian glow and not a feathered selection grow.

We also erode by a pixel before dilating, which removes the fringe of
background colour that survives the cutout and would otherwise be dyed white
by the innermost stroke.

The one wrinkle: a binary alpha has no antialiasing at all, so diagonals come
out as visible stair-steps. The whole stack is therefore built at SUPERSAMPLE
times the working size and scaled down at the end, which restores a one-pixel
antialiased edge without softening the shape. Crisp is the goal; jagged is not.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from .. import brand

# The halo, read outward from the subject: white, pink, blue — the trans flag's
# own order. Strokes are drawn widest first so the narrow ones sit on top.
# Widths are in pixels at THUMB_W and scale with the output size.
DEFAULT_STROKES = [
    {"width": 30, "colour": brand.TRANS_BLUE},
    {"width": 20, "colour": brand.TRANS_PINK},
    {"width": 10, "colour": brand.WHITE},
]

THUMB_W, THUMB_H = 1280, 720

# The text block. Not in brand.py because it is thumbnail furniture rather than
# part of the on-screen identity — the cards must stay trans-flag palette.
THUMB_ACCENT = "#D42B2B"

# How much of the frame height the speaker fills, and how far off the right
# edge they sit. The halo traces the subject, so an oversized subject pushes
# the outline off-frame and loses the sticker read entirely.
SUBJECT_HEIGHT = 0.80
SUBJECT_MARGIN = 0.04

# Drop shadow: hard enough to read as a sticker. A soft shadow undoes the
# crispness the stroke stack just bought.
SHADOW = {"offset": (10, 12), "blur": 5, "opacity": 140, "grow": 4}

# Build the stroke stack at this multiple of the output size, then scale down.
# The alpha has to be hard to keep corners sharp, and hard alpha aliases, so
# the antialiasing is bought back in the downscale instead.
SUPERSAMPLE = 2

# Sampling rate inside a candidate window. Blinks last ~120ms, so 5/s is enough
# to land on open eyes without scoring thousands of frames.
SAMPLE_FPS = 5.0

# Below these the frame is disqualified outright rather than merely scored down.
EYE_OPEN_MIN = 0.19          # eye aspect ratio; a blink drops well under this
MOUTH_OPEN_MAX = 0.62        # mouth aspect ratio; mid-vowel shapes exceed this
SHARPNESS_MIN = 45.0         # variance of Laplacian over the face crop


@dataclass
class Candidate:
    """One scored frame, in both clocks so it can be traced back to the footage."""
    path: str
    raw_time: float
    timeline_time: float
    score: float
    eye: float
    mouth: float
    sharpness: float
    yaw: float
    note: str = ""

    def label(self) -> str:
        m, s = divmod(int(self.raw_time), 60)
        return f"{m}:{s:02d}"


# --- frame extraction -------------------------------------------------------

def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"{tool} not found on PATH")
    return path


def extract_frames(video: str | Path, out_dir: Path, start: float,
                   end: float, fps: float = SAMPLE_FPS) -> list[tuple[float, Path]]:
    """Pull frames from [start, end) as PNGs. Returns [(raw_time, path)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start:.2f}".replace(".", "_")
    pattern = out_dir / f"f{tag}_%04d.png"
    subprocess.run(
        [_require("ffmpeg"), "-nostdin", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
         "-vf", f"fps={fps}", "-fps_mode", "passthrough", str(pattern)],
        check=True, capture_output=True,
    )
    frames = sorted(out_dir.glob(f"f{tag}_*.png"))
    return [(start + i / fps, p) for i, p in enumerate(frames)]


# --- face scoring -----------------------------------------------------------

def _face_mesh():
    try:
        import mediapipe as mp  # noqa: PLC0415  (optional, large)
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is needed to score frames; pip install mediapipe, "
            "or pass --at/--frame to choose a frame yourself"
        ) from exc
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5,
    )


# Landmark indices in MediaPipe's 468-point mesh.
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_MOUTH = (61, 81, 311, 291, 402, 178)
_NOSE, _CHIN, _L_CHEEK, _R_CHEEK = 1, 152, 234, 454


def _aspect_ratio(pts: list[tuple[float, float]]) -> float:
    """
    Eye/mouth aspect ratio: mean vertical opening over horizontal width.

    Same formula for both — six points, corners first and last. Scale-free, so
    it does not care how big the face is in frame.
    """
    (p0, p1, p2, p3, p4, p5) = pts
    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    width = dist(p0, p3)
    if width < 1e-6:
        return 0.0
    return (dist(p1, p5) + dist(p2, p4)) / (2.0 * width)


def score_frame(path: Path, mesh) -> dict | None:
    """Score one frame. None when no face was found."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    result = mesh.process(arr)
    if not result.multi_face_landmarks:
        return None

    h, w = arr.shape[:2]
    lm = result.multi_face_landmarks[0].landmark
    pt = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731

    eye = (_aspect_ratio([pt(i) for i in _LEFT_EYE])
           + _aspect_ratio([pt(i) for i in _RIGHT_EYE])) / 2.0
    mouth = _aspect_ratio([pt(i) for i in _MOUTH])

    # Yaw from the nose's offset between the cheek points: 0 = facing camera.
    lc, rc, nose = pt(_L_CHEEK), pt(_R_CHEEK), pt(_NOSE)
    span = abs(rc[0] - lc[0]) or 1.0
    yaw = abs(((nose[0] - lc[0]) / span) - 0.5) * 2.0

    # Sharpness on the face crop only — a sharp background behind a blurred
    # head would otherwise pass.
    ys = [p[1] for p in (pt(_NOSE), pt(_CHIN), lc, rc)]
    xs = [p[0] for p in (pt(_NOSE), pt(_CHIN), lc, rc)]
    x0, x1 = max(0, int(min(xs)) - 40), min(w, int(max(xs)) + 40)
    y0, y1 = max(0, int(min(ys)) - 80), min(h, int(max(ys)) + 40)
    crop = arr[y0:y1, x0:x1]
    grey = crop.mean(axis=2) if crop.size else np.zeros((2, 2))
    lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2]
           + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1]) if grey.size > 16 else np.zeros((1,))
    sharpness = float(lap.var())

    face_h = abs(pt(_CHIN)[1] - pt(_NOSE)[1]) / h

    return {"eye": eye, "mouth": mouth, "yaw": yaw,
            "sharpness": sharpness, "face_h": face_h}


def _combine(m: dict) -> tuple[float, str]:
    """Fold the measurements into one score, and say why if disqualified."""
    if m["eye"] < EYE_OPEN_MIN:
        return 0.0, "eyes closed / mid-blink"
    if m["mouth"] > MOUTH_OPEN_MAX:
        return 0.0, "mouth open mid-word"
    if m["sharpness"] < SHARPNESS_MIN:
        return 0.0, "motion blur"

    # Past the filters, prefer open eyes, a camera-facing head, a sharp and
    # reasonably large face. Deliberately gentle: these all pass, and the
    # ranking is only a running order for your eye, not a verdict.
    score = (
        min(m["eye"] / 0.30, 1.0) * 0.30
        + (1.0 - min(m["yaw"], 1.0)) * 0.30
        + min(m["sharpness"] / 300.0, 1.0) * 0.25
        + min(m["face_h"] / 0.18, 1.0) * 0.15
    )
    return round(score, 4), ""


# --- choosing where to look -------------------------------------------------

def candidate_windows(timeline, pad: float = 2.0) -> list[tuple[float, float]]:
    """
    Raw-time windows worth searching: around pull-quotes and chapter openings.

    These are the moments the analysis already judged to carry the argument,
    which is a far better proxy for "sets the tone" than anything in the pixels.

    The timeline is conformed by this point, so its events are in timeline time
    and have to be mapped back to raw time to address the camera file.
    """
    marks: list[float] = []
    for ov in timeline.overlays:
        if ov.kind in ("pull_quote", "reaction"):
            marks.append((ov.start + ov.end) / 2.0)
    for ch in timeline.chapters:
        marks.append(ch.start + 1.0)

    windows = []
    for t in marks:
        mapped = timeline.timeline_to_raw(t)
        if mapped is None:
            continue
        _, raw = mapped
        windows.append((max(0.0, raw - pad), raw + pad))
    return _merge(sorted(windows))


def _merge(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in windows:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def propose(timeline, video: str | Path, work_dir: Path, top: int = 10,
            windows: list[tuple[float, float]] | None = None,
            remote: bool = False) -> list[Candidate]:
    """
    Score frames in the candidate windows and write a contact sheet.

    Returns the shortlist, best first. Nothing is chosen — that is yours.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)

    windows = windows or candidate_windows(timeline)
    if not windows:
        raise RuntimeError(
            "no candidate windows — run the analyse stage first, or pass --at"
        )

    # Frame extraction is ffmpeg's job either way; only the scoring moves.
    harvested: list[tuple[float, Path]] = []
    for start, end in windows:
        harvested.extend(extract_frames(video, frames_dir, start, end))

    if remote:
        measurements = _score_remotely(harvested)
    else:
        mesh = _face_mesh()
        measurements = {p: score_frame(p, mesh) for _, p in harvested}

    scored: list[Candidate] = []
    for raw_time, path in harvested:
        measured = measurements.get(path)
        if measured is None:
            continue                          # no face found in this frame
        score, note = _combine(measured)
        if score <= 0.0:
            continue                          # disqualified, not merely weak
        tl_time = timeline.raw_to_timeline(raw_time)
        scored.append(Candidate(
            path=str(path), raw_time=raw_time,
            timeline_time=tl_time if tl_time is not None else -1.0,
            score=score, eye=round(measured["eye"], 4),
            mouth=round(measured["mouth"], 4),
            sharpness=round(measured["sharpness"], 1),
            yaw=round(measured["yaw"], 4), note=note,
        ))

    scored.sort(key=lambda c: c.score, reverse=True)
    shortlist = _spread(scored, top)
    (work_dir / "candidates.json").write_text(
        json.dumps([asdict(c) for c in shortlist], indent=2), encoding="utf-8")
    if shortlist:
        contact_sheet(shortlist, work_dir / "contact-sheet.png")
    return shortlist


def _spread(scored: list[Candidate], top: int, min_gap: float = 1.5
            ) -> list[Candidate]:
    """
    Take the best `top`, but never two frames from within `min_gap` seconds.

    Without this the shortlist is ten near-identical frames from whichever
    single moment happened to be well lit, which is no choice at all.
    """
    picked: list[Candidate] = []
    for cand in scored:
        if all(abs(cand.raw_time - p.raw_time) >= min_gap for p in picked):
            picked.append(cand)
        if len(picked) >= top:
            break
    return picked


def contact_sheet(candidates: list[Candidate], out_path: Path,
                  cols: int = 5, cell_w: int = 480) -> Path:
    """A numbered grid of the shortlist, so the choice takes seconds."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    thumbs = []
    for cand in candidates:
        img = Image.open(cand.path).convert("RGB")
        cell_h = int(cell_w * img.height / img.width)
        thumbs.append(img.resize((cell_w, cell_h), Image.LANCZOS))

    cell_h = max(t.height for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    bar = 34
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + bar)),
                      brand.rgba(brand.WARM_DARK)[:3])
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(brand.font_path(), 22)
    except Exception:                                   # noqa: BLE001
        font = ImageFont.load_default()

    for i, (thumb, cand) in enumerate(zip(thumbs, candidates)):
        x, y = (i % cols) * cell_w, (i // cols) * (cell_h + bar)
        sheet.paste(thumb, (x, y))
        draw.text((x + 8, y + cell_h + 6),
                  f"[{i}] {cand.label()}  score {cand.score:.2f}",
                  fill=brand.rgba(brand.WHITE)[:3], font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _score_remotely(harvested: list[tuple[float, Path]]) -> dict:
    """
    Batch the frames to a GPU worker and index the results back by path.

    One job for all frames: per-frame jobs would pay the worker's cold start
    dozens of times over, which is slower than simply scoring locally.
    """
    from .. import remote as remote_mod

    paths = [p for _, p in harvested]
    by_name = {p.name: p for p in paths}
    out: dict = {}
    for row in remote_mod.score_frames(paths):
        path = by_name.get(row.get("name", ""))
        if path is None:
            continue
        out[path] = None if not row.get("face") else {
            k: row[k] for k in ("eye", "mouth", "yaw", "sharpness", "face_h")
        }
    return out


# --- the sticker halo -------------------------------------------------------

def cutout(frame: Path, out_path: Path, remote: bool = False) -> Path:
    """Remove the background. Output is RGBA with a soft edge; halo() hardens it."""
    if remote:
        from .. import remote as remote_mod
        return remote_mod.cutout(frame, out_path)
    try:
        from rembg import remove  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("rembg is needed for the cutout; pip install rembg") from exc
    from PIL import Image  # noqa: PLC0415

    result = remove(Image.open(frame).convert("RGBA"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    return out_path


def _binary_alpha(img, threshold: int = 128, erode: int = 1):
    """
    Harden a soft cutout alpha to 0/255, then erode by `erode` px.

    The threshold is what makes the outline crisp instead of rounded. The erode
    strips the halo of leftover background colour around the subject, which the
    innermost stroke would otherwise light up.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415

    alpha = img.getchannel("A").point(lambda v: 255 if v >= threshold else 0)
    for _ in range(max(0, erode)):
        alpha = alpha.filter(ImageFilter.MinFilter(3))
    return Image.merge("RGBA", (*img.split()[:3], alpha))


def _dilate(mask, radius: int):
    """
    Grow a binary mask by `radius` px.

    MaxFilter is a square structuring element, so it holds sharp corners rather
    than rounding them — which is the whole point, and the reason this does not
    use a blur or a feathered selection grow.
    """
    from PIL import ImageFilter  # noqa: PLC0415

    grown = mask
    remaining = radius
    while remaining > 0:
        step = min(remaining, 5)                # MaxFilter caps at size 11
        grown = grown.filter(ImageFilter.MaxFilter(step * 2 + 1))
        remaining -= step
    return grown


def halo(cut_png: Path, out_path: Path, strokes: list[dict] | None = None,
         shadow: dict | None = None, scale: float = 1.0,
         supersample: int = SUPERSAMPLE) -> Path:
    """
    Add the stroke stack and drop shadow to a transparent cutout.

    Output stays transparent outside the sticker, so it drops straight into a
    template. Strokes are drawn widest first; the subject goes on last.

    Everything is composed at `supersample` times size and scaled back down, so
    the edge is antialiased without the shape ever being blurred. Pass
    supersample=1 to inspect the raw binary geometry.
    """
    from PIL import Image, ImageFilter  # noqa: PLC0415

    strokes = strokes or DEFAULT_STROKES
    shadow = SHADOW if shadow is None else shadow
    ss = max(1, int(supersample))

    source = Image.open(cut_png).convert("RGBA")
    if ss > 1:
        source = source.resize((source.width * ss, source.height * ss),
                               Image.LANCZOS)
    # Threshold *after* the upscale: resampling reintroduces soft pixels, and a
    # soft matte is exactly what rounds the corners off.
    subject = _binary_alpha(source, erode=ss)
    mask = subject.getchannel("A")

    widest = max(int(s["width"] * scale * ss) for s in strokes)
    grow = shadow["grow"] * ss
    blur = shadow["blur"] * ss
    offset = (shadow["offset"][0] * ss, shadow["offset"][1] * ss)
    pad = widest + grow + max(abs(v) for v in offset) + blur + 4 * ss
    size = (subject.width + pad * 2, subject.height + pad * 2)

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))

    # Shadow first, cast from the outermost stroke so it sits behind everything.
    if shadow["opacity"] > 0:
        shadow_mask = _dilate(mask, widest + grow)
        if blur:
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(blur))
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        tinted = Image.new("RGBA", subject.size, (0, 0, 0, shadow["opacity"]))
        layer.paste(tinted, (pad + offset[0], pad + offset[1]), shadow_mask)
        canvas = Image.alpha_composite(canvas, layer)

    for stroke in sorted(strokes, key=lambda s: -s["width"]):
        grown = _dilate(mask, int(stroke["width"] * scale * ss))
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        fill = Image.new("RGBA", subject.size, brand.rgba(stroke["colour"]))
        layer.paste(fill, (pad, pad), grown)
        canvas = Image.alpha_composite(canvas, layer)

    top = Image.new("RGBA", size, (0, 0, 0, 0))
    top.paste(subject, (pad, pad), subject)
    canvas = Image.alpha_composite(canvas, top)

    if ss > 1:
        canvas = canvas.resize((canvas.width // ss, canvas.height // ss),
                               Image.LANCZOS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


# --- the template -----------------------------------------------------------

def compose(sticker: Path, out_path: Path, phrase: str = "",
            template: Path | None = None,
            width: int = THUMB_W, height: int = THUMB_H) -> Path:
    """
    Lay the halo'd subject into the thumbnail template with the text block.

    `template` is an optional background PNG (your existing artwork). Without
    one we paint a flat brand background so the stage still produces something
    usable on a fresh machine.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    if template and Path(template).exists():
        bg = Image.open(template).convert("RGBA").resize((width, height), Image.LANCZOS)
    else:
        bg = Image.new("RGBA", (width, height), brand.rgba(brand.WARM_DARK))

    sub = Image.open(sticker).convert("RGBA")
    # The halo hugs the speakers; it is not a border around the whole frame.
    # So the subject is sized to a fraction of the frame and sits on the
    # bottom-right, leaving the left for the text block and keeping the whole
    # outline visible rather than running off the edges.
    target_h = int(height * SUBJECT_HEIGHT)
    sub = sub.resize((max(1, int(sub.width * target_h / sub.height)), target_h),
                     Image.LANCZOS)
    x = width - sub.width - int(width * SUBJECT_MARGIN)
    bg.alpha_composite(sub, (x, height - sub.height))

    if phrase:
        _text_block(bg, phrase.upper(), width, height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(out_path, quality=95)
    return out_path


def _text_block(img, text: str, width: int, height: int) -> None:
    """The red block and its label, sized to the phrase rather than fixed."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    draw = ImageDraw.Draw(img)
    size = 116 if len(text) <= 12 else 92 if len(text) <= 18 else 74
    try:
        font = ImageFont.truetype(brand.font_path(weight="Bold"), size)
    except Exception:                                   # noqa: BLE001
        font = ImageFont.load_default()

    words, lines, current = text.split(), [], ""
    limit = int(width * 0.46)
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= limit or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)

    pad_x, pad_y, gap = 30, 18, 12
    heights = [size + gap] * len(lines)
    block_h = sum(heights) + pad_y * 2 - gap
    y = int(height * 0.5 - block_h / 2)

    for line in lines:
        w = draw.textlength(line, font=font)
        box = [int(width * 0.05), y,
               int(width * 0.05) + int(w) + pad_x * 2, y + size + pad_y * 2]
        draw.rectangle(box, fill=brand.rgba(THUMB_ACCENT))
        draw.text((box[0] + pad_x, y + pad_y), line,
                  font=font, fill=brand.rgba(brand.WHITE))
        y += size + pad_y * 2 + gap



def build(frame: Path, work_dir: Path, out_path: Path, phrase: str = "",
          template: Path | None = None, remote: bool = False) -> Path:
    """Chosen frame → cutout → halo → template. The deterministic half."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cut = cutout(frame, work_dir / "cutout.png", remote=remote)
    sticker = halo(cut, work_dir / "sticker.png")
    return compose(sticker, out_path, phrase=phrase, template=template)
