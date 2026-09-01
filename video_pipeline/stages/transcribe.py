"""
transcribe.py — speech to a speaker-labelled, timestamped transcript.

Reuses the workspace's shared media_transcribe.py (faster-whisper) so this
pipeline and Jennifer's transcription skill stay on one implementation.

Diarization is deliberately pluggable. On a single mixed track with two people
who interrupt each other, diarization is *advisory*: it is good enough to draft
name-card timing and to label a transcript for the LLM, and not good enough to
trust blind. Anything it is unsure about becomes a review item.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from ..edl import ReviewItem, Subtitle

# An outside transcriber to reuse instead of loading our own model, if the host
# machine has one. Optional: unset, we call faster-whisper directly.
HOST_TRANSCRIBER = os.environ.get("MEDIA_TRANSCRIBE_PATH", "")

# Below this diarization confidence we ask a human rather than guess on air.
DIARIZATION_CONFIDENCE_FLOOR = 0.55


def _load_host_transcriber():
    """Reuse a host machine's transcriber when MEDIA_TRANSCRIBE_PATH names one."""
    if not HOST_TRANSCRIBER or not Path(HOST_TRANSCRIBER).exists():
        return None
    spec = importlib.util.spec_from_file_location("media_transcribe", HOST_TRANSCRIBER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module if hasattr(module, "transcribe_file") else None


def transcribe(audio_path: str | Path, model: str = "base",
               device: str = "cpu", language: str = "en",
               remote: bool = False) -> dict:
    """
    Return a result dict of segments with start/end/text.

    faster-whisper is a generator API, so we materialise it here — the analysis
    stage needs the whole transcript at once to reason about structure.
    """
    if remote:
        from .. import remote as remote_mod
        # large-v3 on a rented GPU: the reason to go remote is to stop trading
        # accuracy for a CPU's patience, so the local default is not reused.
        return remote_mod.transcribe(
            Path(audio_path),
            model="large-v3" if model in ("base", "small", "medium") else model,
            language=language,
        )

    host = _load_host_transcriber()
    if host is not None:
        return host.transcribe_file(str(audio_path), model=model,
                                    device=device, language=language)

    from faster_whisper import WhisperModel  # noqa: PLC0415  (heavy import)

    compute_type = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel(model, device=device, compute_type=compute_type)
    segments, info = whisper.transcribe(
        str(audio_path), language=language, vad_filter=True,
        word_timestamps=True,
    )
    out = []
    for seg in segments:
        out.append({
            "start": seg.start, "end": seg.end, "text": seg.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word}
                for w in (seg.words or [])
            ],
        })
    return {"segments": out, "language": info.language,
            "text": " ".join(s["text"] for s in out)}


def diarize(audio_path: str | Path, num_speakers: int = 2) -> list[dict]:
    """
    Return [{start, end, speaker, confidence}] using pyannote if available.

    Returns [] when pyannote is not installed, which is the normal case on this
    CPU box — the caller then falls back to unlabelled segments and flags them.
    Set HF_TOKEN for the pyannote model download.
    """
    if importlib.util.find_spec("pyannote.audio") is None:
        return []
    from pyannote.audio import Pipeline  # noqa: PLC0415  (optional dependency)

    token = os.environ.get("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token,
    )
    annotation = pipeline(str(audio_path), num_speakers=num_speakers)
    return [
        {"start": turn.start, "end": turn.end, "speaker": label, "confidence": 1.0}
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]


def _speaker_at(turns: list[dict], start: float, end: float) -> tuple[str, float]:
    """
    Label a transcript segment with whichever speaker turn overlaps it most.

    Confidence is the winning turn's share of the segment, so a segment split
    evenly between two people scores ~0.5 and gets flagged for review.
    """
    if not turns:
        return "", 0.0
    totals: dict[str, float] = {}
    for t in turns:
        overlap = min(end, t["end"]) - max(start, t["start"])
        if overlap > 0:
            totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + overlap
    if not totals:
        return "", 0.0
    span = max(end - start, 1e-6)
    winner = max(totals, key=totals.get)
    return winner, min(totals[winner] / span, 1.0)


def run(audio_path: str | Path, speaker_names: dict | None = None,
        model: str = "base", device: str = "cpu", remote: bool = False,
        ) -> tuple[list[Subtitle], list[ReviewItem], dict]:
    """
    Produce subtitles in raw time, plus review items for anything uncertain.

    `speaker_names` maps diarization labels to real names, e.g.
    {"SPEAKER_00": "Heather", "SPEAKER_01": "Sophie"}.
    """
    speaker_names = speaker_names or {}
    result = transcribe(audio_path, model=model, device=device, remote=remote)
    # A remote worker diarizes in the same job when it has a token, so only
    # fall back to a local pass when it returned nothing.
    turns = result.get("speakers") or diarize(audio_path)

    subtitles: list[Subtitle] = []
    review: list[ReviewItem] = []

    if not turns:
        review.append(ReviewItem(
            kind="diarization", at=0.0, severity="check",
            note="pyannote not installed — transcript has no speaker labels; "
                 "name cards and per-speaker punch-ins are disabled for this episode",
        ))

    for seg in result.get("segments", []):
        label, confidence = _speaker_at(turns, seg["start"], seg["end"])
        name = speaker_names.get(label, label)
        subtitles.append(Subtitle(
            start=float(seg["start"]), end=float(seg["end"]),
            text=seg["text"].strip(), speaker=name,
        ))
        if turns and confidence < DIARIZATION_CONFIDENCE_FLOOR:
            review.append(ReviewItem(
                kind="diarization", at=float(seg["start"]), severity="check",
                note=f"unclear who is speaking ({confidence:.0%} confidence): "
                     f"{seg['text'].strip()[:80]!r}",
            ))

    return subtitles, review, {"language": result.get("language", ""),
                               "diarized": bool(turns)}


def to_srt(subtitles: list[Subtitle]) -> str:
    """SRT for the subtitle track and for the title/description prompt."""
    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, sub in enumerate(subtitles, 1):
        blocks.append(f"{i}\n{stamp(sub.start)} --> {stamp(sub.end)}\n{sub.text}\n")
    return "\n".join(blocks)


def to_prompt_transcript(subtitles: list[Subtitle]) -> str:
    """Compact speaker-labelled transcript for the LLM analysis stage."""
    lines = []
    for sub in subtitles:
        who = sub.speaker or "UNKNOWN"
        lines.append(f"[{sub.start:.1f}] {who}: {sub.text}")
    return "\n".join(lines)
