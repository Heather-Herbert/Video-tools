"""
analyse.py — turn a speaker-labelled transcript into edit decisions.

The LLM never touches media and never writes editor XML. It reads a transcript
and returns JSON in raw source time; everything structural (deciding the actual
cut list, mapping to the timeline, laying out tracks) is ordinary code, because
that part must be deterministic and reviewable.

Backend order is Claude CLI → agy → DeepSeek → OpenRouter, via video_pipeline.llm.
"""

from __future__ import annotations

from .. import llm, voice
from ..edl import (
    Chapter, Cut, NEEDS_REVIEW, Overlay, Reframe, ReviewItem, Subtitle, Timeline,
)

# Keep this much air either side of kept speech so cuts don't clip breaths.
CUT_PAD = 0.18
# Gaps shorter than this are left alone — removing them sounds unnatural.
MIN_GAP = 0.60
# A deliberate beat before a punchline. The LLM can protect a gap up to here.
MAX_PROTECTED_BEAT = 1.60


ANALYSIS_PROMPT = """\
You are assisting the edit of a two-person UK news-commentary video. The hosts
are Heather and Sophie. It covers UK politics and lived experience.

{voice}

Below is a timestamped, speaker-labelled transcript. Timestamps are seconds
into the raw recording. Analyse it and return JSON only.

Return this exact shape:

{
  "protected_beats": [{"at": 12.4, "why": "deliberate pause before punchline"}],
  "drop_spans":      [{"start": 30.1, "end": 41.8, "why": "false start, retold better after"}],
  "chapters":        [{"at": 0.0, "title": "Short searchable topic title"}],
  "pull_quotes":     [{"at": 88.2, "until": 93.0, "quote": "verbatim line", "speaker": "Sophie"}],
  "citations":       [{"at": 120.0, "until": 126.0, "claim": "what was asserted",
                       "outlet": "outlet if named, else empty", "headline": "", "date": ""}],
  "stats":           [{"at": 140.0, "until": 146.0, "value": "84%", "label": "what it measures",
                       "source": "source if named, else empty"}],
  "reactions":       [{"at": 160.0, "until": 162.0, "text": "Wait, what?"}],
  "reframes":        [{"start": 200.0, "end": 214.0, "target": "heather"}],
  "ambiguous":       [{"at": 250.0, "why": "unclear whether this is a joke or a claim"}]
}

Rules:
- Quote verbatim in "quote"; never invent a line that is not in the transcript.
- Only flag a citation where a specific factual or news claim is made. Leave
  "outlet"/"headline" empty rather than guessing a source. A human verifies these.
- "drop_spans" are for false starts and abandoned tangents only. Do not drop
  content because you disagree with it.
- "reframes" punch in on one speaker during a sustained solo stretch (10s+).
  Use "heather", "sophie" or "both".
- 4–8 chapters for a 10–12 minute video. Titles are searchable, not clever.
- Anything you are unsure about goes in "ambiguous" rather than being acted on.

Transcript:
---
{transcript}
---
"""


def _call_llm(transcript: str, client=None) -> dict:
    prompt = (ANALYSIS_PROMPT
              .replace("{voice}", voice.prompt_block())
              .replace("{transcript}", transcript))
    return llm.classify_json(prompt, client)


def build_cuts(subtitles: list[Subtitle], source_id: str, duration: float,
               protected: list[float], drops: list[tuple[float, float]],
               ) -> list[Cut]:
    """
    Build the kept-segment list from speech spans.

    Speech is what the transcriber found; everything between spans is a gap.
    A gap is removed when it is longer than MIN_GAP and not protected as a
    deliberate beat. Protected beats are preserved, but capped, so a protected
    twelve-second silence still gets trimmed to a usable pause.
    """
    spans = [(s.start, s.end) for s in subtitles if s.end > s.start]
    spans.sort()
    if not spans:
        return []

    def is_dropped(a: float, b: float) -> bool:
        mid = (a + b) / 2
        return any(d0 <= mid <= d1 for d0, d1 in drops)

    kept: list[list[float]] = []
    for start, end in spans:
        if is_dropped(start, end):
            continue
        start = max(0.0, start - CUT_PAD)
        end = min(duration, end + CUT_PAD) if duration else end + CUT_PAD

        if not kept:
            kept.append([start, end])
            continue

        gap = start - kept[-1][1]
        beat = any(kept[-1][1] - 0.3 <= p <= start + 0.3 for p in protected)
        if gap <= MIN_GAP:
            kept[-1][1] = max(kept[-1][1], end)       # merge, gap too short to cut
        elif beat:
            # Keep the beat, but no longer than a beat needs to be.
            kept[-1][1] = min(start, kept[-1][1] + MAX_PROTECTED_BEAT)
            kept.append([start, end])
        else:
            kept.append([start, end])

    return [
        Cut(src=source_id, start=round(a, 3), end=round(b, 3),
            reason="dead air removed" if i else "head trimmed")
        for i, (a, b) in enumerate(kept)
    ]


def _overlay(kind: str, at: float, until: float, fields: dict,
             min_len: float = 2.5) -> Overlay:
    """Cards need a readable dwell time; extend anything the LLM timed too tight."""
    until = max(until, at + min_len)
    return Overlay(start=float(at), end=float(until), kind=kind, fields=fields)


def run(timeline: Timeline, subtitles: list[Subtitle], client=None,
        transcript: str | None = None) -> Timeline:
    """
    Populate the timeline with cuts, overlays, chapters and review items.

    Everything produced here is still in raw time. Call timeline.conform()
    afterwards to move it onto the edited timeline.
    """
    from .transcribe import to_prompt_transcript

    timeline.subtitles = subtitles
    transcript = transcript or to_prompt_transcript(subtitles)
    analysis = _call_llm(transcript, client)

    protected = [float(b["at"]) for b in analysis.get("protected_beats", [])]
    drops = [(float(d["start"]), float(d["end"])) for d in analysis.get("drop_spans", [])]

    main = next((s for s in timeline.sources if s.role == "main"), None)
    if main is None:
        raise ValueError("timeline has no source with role 'main'")

    timeline.cuts = build_cuts(subtitles, main.id, main.duration, protected, drops)
    if not timeline.cuts:
        raise RuntimeError("analysis produced no cuts — transcript was empty?")

    for d in analysis.get("drop_spans", []):
        timeline.review.append(ReviewItem(
            kind="ambiguous_cut", at=float(d["start"]), severity="check",
            note=f"dropped {float(d['end']) - float(d['start']):.1f}s: {d.get('why', '')}",
        ))

    # --- chapters, and a title card for each -------------------------------
    for i, ch in enumerate(analysis.get("chapters", []), 1):
        at = float(ch["at"])
        timeline.chapters.append(Chapter(start=at, title=ch["title"]))
        if i > 1:  # the first chapter is the video's own opening, not a break
            timeline.overlays.append(_overlay(
                "chapter", at, at + 3.5,
                {"title": ch["title"], "index": f"Chapter {i}"},
            ))

    # --- name cards on each speaker's first appearance ---------------------
    seen: set[str] = set()
    for sub in subtitles:
        if sub.speaker and sub.speaker not in seen:
            seen.add(sub.speaker)
            timeline.overlays.append(_overlay(
                "name_card", sub.start, sub.start + 4.0,
                {"name": sub.speaker, "role": "The Polycule"},
            ))

    for q in analysis.get("pull_quotes", []):
        timeline.overlays.append(_overlay(
            "pull_quote", float(q["at"]), float(q.get("until", 0)),
            {"quote": q["quote"], "speaker": q.get("speaker", "")}, min_len=3.0,
        ))

    for c in analysis.get("citations", []):
        at = float(c["at"])
        timeline.overlays.append(_overlay(
            "citation", at, float(c.get("until", 0)),
            {"outlet": c.get("outlet", ""), "headline": c.get("headline", ""),
             "date": c.get("date", "")}, min_len=4.0,
        ))
        # Blocking: a source card must never render on an unverified claim.
        timeline.review.append(ReviewItem(
            kind="citation", at=at, severity="block",
            note=f"verify and complete source card for claim: {c.get('claim', '')[:120]}",
        ))

    for s in analysis.get("stats", []):
        at = float(s["at"])
        timeline.overlays.append(_overlay(
            "stat", at, float(s.get("until", 0)),
            {"value": s.get("value", ""), "label": s.get("label", ""),
             "source": s.get("source", "")}, min_len=3.5,
        ))
        timeline.review.append(ReviewItem(
            kind="stat", at=at, severity="block",
            note=f"verify figure {s.get('value', '')} — {s.get('label', '')}",
        ))

    for r in analysis.get("reactions", []):
        timeline.overlays.append(_overlay(
            "reaction", float(r["at"]), float(r.get("until", 0)),
            {"text": r.get("text", "")}, min_len=1.5,
        ))

    for rf in analysis.get("reframes", []):
        target = rf.get("target", "both").lower()
        # Punch in toward the speaker's side of a two-shot. Camera is static,
        # so a fixed left/right bias is a safe default; tune per set-up.
        x = {"heather": 0.34, "sophie": 0.66}.get(target, 0.5)
        timeline.reframes.append(Reframe(
            start=float(rf["start"]), end=float(rf["end"]),
            target=target, scale=1.0 if target == "both" else 1.35, x=x, y=0.46,
        ))

    for a in analysis.get("ambiguous", []):
        timeline.review.append(ReviewItem(
            kind="ambiguous_cut", at=float(a["at"]), severity="check",
            note=a.get("why", ""),
        ))

    timeline.overlays.sort(key=lambda o: o.start)
    timeline.review.sort(key=lambda r: r.at)
    return timeline


def youtube_chapters(timeline: Timeline) -> str:
    """Chapter timestamps for the YouTube description, in timeline time."""
    lines = []
    for ch in sorted(timeline.chapters, key=lambda c: c.start):
        m, s = divmod(int(ch.start), 60)
        h, m = divmod(m, 60)
        stamp = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        lines.append(f"{stamp} {ch.title}")
    # YouTube requires the first chapter to start at zero.
    if lines and not lines[0].startswith(("0:00", "0:00:00")):
        lines.insert(0, "0:00 Intro")
    return "\n".join(lines)
