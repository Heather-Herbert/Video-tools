"""
render.py — loudness normalisation and headless render.

Rendering is MLT's job (`melt`), which is CPU-bound; there is nothing here for
a GPU to do. Loudness is a two-pass ffmpeg `loudnorm` so the measured values
drive the second pass — single-pass loudnorm drifts on dialogue.

YouTube normalises to about -14 LUFS, so we target that and leave 1.5 dB of
true-peak headroom for the codec.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from ..edl import Timeline

TARGET_LUFS = -14.0
TARGET_TP = -1.5
TARGET_LRA = 11.0


class RenderError(RuntimeError):
    pass


def measure_loudness(path: str | Path) -> dict:
    """First loudnorm pass: measure. Returns the JSON stats block."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.S)
    if not match:
        raise RenderError(f"could not measure loudness of {path}")
    return json.loads(match.group(0))


def normalise(src: str | Path, out: str | Path) -> Path:
    """Second loudnorm pass: apply, using the measured values."""
    src, out = Path(src), Path(out)
    stats = measure_loudness(src)
    filt = (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
        f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}:linear=true:print_format=summary"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-af", filt, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out)],
        check=True,
    )
    return out


def _render_target(project: Path, fps: int) -> tuple[Path, int]:
    """
    Prepare a melt-renderable copy of the project.

    Two things melt needs that a .kdenlive file does not carry:

    1. The root <mlt> element points at `main_bin`, so melt would render the
       project bin — every clip end to end — rather than the timeline. We
       repoint it at the master tractor.
    2. `black_track` is 2^31 frames long, so without an explicit frame range
       melt keeps rendering black long past the end of the edit.

    Returns the temp project path and the last frame to render.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(project)
    root = tree.getroot()
    master = next(
        (t for t in root.findall("tractor")
         if any(pr.get("name") == "kdenlive:projectTractor"
                for pr in t.findall("property"))),
        None,
    )
    if master is None:
        raise RenderError(
            f"{project} has no master tractor — was it written by this pipeline?"
        )
    root.set("producer", master.get("id"))

    out_tc = master.get("out") or "00:00:00.000"
    h, m, rest = out_tc.split(":")
    sec, _, ms = rest.partition(".")
    total = int(h) * 3600 + int(m) * 60 + int(sec) + int(ms or 0) / 1000
    last_frame = max(0, round(total * fps))

    target = project.with_suffix(".render.mlt")
    tree.write(target, encoding="unicode")
    return target, last_frame


def render_project(project: str | Path, out: str | Path, fps: int = 25,
                   crf: int = 19, preset: str = "medium",
                   timeout: int = 7200) -> Path:
    """Render a .kdenlive project headlessly via melt."""
    project, out = Path(project), Path(out)
    if not shutil.which("melt"):
        raise RenderError(
            "melt is not on PATH — install the MLT tools (apt install melt) "
            "or open the project in Kdenlive and render from there"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    target, last_frame = _render_target(project, fps)
    try:
        proc = subprocess.run(
            ["melt", "-progress", str(target), "in=0", f"out={last_frame}",
             "-consumer", f"avformat:{out}",
             "vcodec=libx264", f"crf={crf}", f"preset={preset}",
             "acodec=aac", "ab=192k", "movflags=+faststart"],
            capture_output=True, text=True, timeout=timeout,
        )
    finally:
        target.unlink(missing_ok=True)
    if proc.returncode != 0 or not out.exists():
        raise RenderError(f"melt failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return out


def run(timeline: Timeline, project: str | Path, out_dir: str | Path,
        skip_review_gate: bool = False) -> Path:
    """
    Render the episode, refusing while blocking review items are outstanding.

    The gate is the point of the human checkpoint: an unverified citation card
    must not reach a rendered file that someone could upload by accident.
    """
    blocking = timeline.blocking_review()
    if blocking and not skip_review_gate:
        lines = "\n".join(f"  [{b.kind} @ {b.at:.1f}s] {b.note}" for b in blocking)
        raise RenderError(
            f"{len(blocking)} blocking review item(s) unresolved — "
            f"resolve them or pass skip_review_gate=True:\n{lines}"
        )

    out_dir = Path(out_dir)
    raw_out = out_dir / f"{timeline.episode.slug}_raw.mp4"
    final = out_dir / f"{timeline.episode.slug}.mp4"
    render_project(project, raw_out, fps=timeline.episode.fps)
    normalise(raw_out, final)
    raw_out.unlink(missing_ok=True)
    return final
