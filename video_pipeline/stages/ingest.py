"""
ingest.py — probe the raw footage and make an editing proxy.

Kdenlive on a CPU-only box struggles to scrub 4K H.264 straight off the camera.
We cut against a small proxy and let the render stage relink to the original,
so quality is never lost to the proxy.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..edl import Source

PROXY_HEIGHT = 720
PROXY_CRF = 23


class MediaError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise MediaError(f"{tool} is not on PATH — install it before running this stage")
    return path


def probe(path: str | Path) -> dict:
    """Return the ffprobe summary for a media file."""
    path = Path(path)
    if not path.exists():
        raise MediaError(f"no such media file: {path}")
    _require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    return {
        "path": str(path.resolve()),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
        "fps": _fps(video) if video else 0.0,
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def _fps(stream: dict) -> float:
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    den = float(den or 1)
    return float(num) / den if den else 0.0


def make_proxy(src: str | Path, out_dir: str | Path,
               height: int = PROXY_HEIGHT) -> Path:
    """
    Transcode to a scrub-friendly proxy. Skipped if the source is already
    small enough — no point spending CPU to make a 720p copy of a 720p file.
    """
    src, out_dir = Path(src), Path(out_dir)
    info = probe(src)
    if info["height"] and info["height"] <= height:
        return src

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_proxy.mp4"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    _require("ffmpeg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vf", f"scale=-2:{height}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(PROXY_CRF),
         "-c:a", "aac", "-b:a", "160k",
         str(out)],
        check=True,
    )
    return out


def extract_audio(src: str | Path, out_dir: str | Path) -> Path:
    """Pull a 16 kHz mono WAV — what the transcriber wants, not what the edit wants."""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}_16k.wav"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    _require("ffmpeg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-vn", str(out)],
        check=True,
    )
    return out


def detect_silence(src: str | Path, noise_db: float = -32.0,
                   min_len: float = 0.55) -> list[tuple[float, float]]:
    """
    Return (start, end) spans of silence via ffmpeg's silencedetect.

    This is the cheap, deterministic half of cut detection. The LLM pass refines
    it — silence alone would cut the deliberate beat before a punchline.
    """
    _require("ffmpeg")
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_len}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    spans, start = [], None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            spans.append((start, end))
            start = None
    return spans


def run(raw_path: str | Path, work_dir: str | Path,
        make_proxies: bool = True) -> tuple[Source, Path, Path]:
    """
    Ingest one camera file.

    Returns (source, proxy_path, audio_path). The Source carries the *original*
    path, because that is what the final render must relink to.
    """
    raw_path, work_dir = Path(raw_path), Path(work_dir)
    info = probe(raw_path)
    source = Source(
        id="main", path=info["path"], role="main",
        duration=info["duration"],
        has_video=info["has_video"], has_audio=info["has_audio"],
    )
    if not info["has_audio"]:
        raise MediaError(f"{raw_path} has no audio track — nothing to transcribe")

    proxy = make_proxy(raw_path, work_dir / "proxy") if make_proxies else raw_path
    audio = extract_audio(raw_path, work_dir / "audio")
    return source, proxy, audio
