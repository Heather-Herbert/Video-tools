#!/usr/bin/env python3
"""
run.py — drive the episode pipeline, one resumable stage at a time.

    python -m video_pipeline.run ingest      --raw footage.mp4 --slug 2026-09-06
    python -m video_pipeline.run transcribe  --slug 2026-09-06
    python -m video_pipeline.run analyse     --slug 2026-09-06
    python -m video_pipeline.run graphics    --slug 2026-09-06
    python -m video_pipeline.run thumbnail   --slug 2026-09-06 [--pick 3]
    python -m video_pipeline.run metadata    --slug 2026-09-06
    python -m video_pipeline.run assemble    --slug 2026-09-06
    python -m video_pipeline.run review      --slug 2026-09-06
    python -m video_pipeline.run render      --slug 2026-09-06
    python -m video_pipeline.run all         --raw footage.mp4 --slug 2026-09-06

Every stage reads and writes `<episodes>/<slug>/<slug>.edl.json`, so a stage
can be re-run on its own after a fix without redoing the expensive ones. The
transcript is the slow step; the analysis is the one you will iterate on.

`render` refuses while blocking review items are open. That gate is the point:
an unverified citation must not reach a file someone could upload by mistake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import llm, remote
from .edl import Episode, ReviewItem, Source, Subtitle, Timeline
from .stages import (
    analyse, graphics, ingest, metadata as metadata_stage,
    render as render_stage, thumbnail as thumbnail_stage, transcribe,
)
from .writers import kdenlive

llm.load_env()

DEFAULT_ROOT = Path(os.environ.get("POLYCULE_EPISODES", Path.home() / "episodes"))
ASSETS = Path(__file__).resolve().parents[1] / "assets"

SPEAKER_NAMES = {"SPEAKER_00": "Heather", "SPEAKER_01": "Sophie"}


class Workspace:
    """Paths for one episode. Everything lives under a single slug directory."""

    def __init__(self, slug: str, root: Path = DEFAULT_ROOT):
        self.slug = slug
        self.dir = Path(root) / slug
        self.work = self.dir / "work"
        self.graphics = self.work / "graphics"
        self.output = self.dir / "output"
        self.edl = self.dir / f"{slug}.edl.json"
        self.project = self.dir / f"{slug}.kdenlive"
        self.srt = self.dir / f"{slug}.srt"
        self.transcript = self.work / "transcript.json"
        self.thumbnail = self.dir / f"{slug}-thumb.png"
        self.metadata = self.dir / f"{slug}-metadata.txt"
        self.thumb_work = self.work / "thumbnail"

    def ensure(self):
        for d in (self.dir, self.work, self.graphics, self.output, self.thumb_work):
            d.mkdir(parents=True, exist_ok=True)

    def load(self) -> Timeline:
        if not self.edl.exists():
            raise SystemExit(
                f"no timeline at {self.edl} — run the earlier stages first"
            )
        return Timeline.load(self.edl)


def _openrouter_client():
    """Optional final fallback for the LLM chain; None if no key is configured."""
    return llm.openrouter_client()


# --- stages -----------------------------------------------------------------

def stage_ingest(ws: Workspace, args) -> Timeline:
    ws.ensure()
    source, proxy, audio = ingest.run(args.raw, ws.work, make_proxies=not args.no_proxy)
    tl = Timeline(episode=Episode(
        slug=ws.slug, title=args.title or ws.slug,
        recorded=args.recorded or "", fps=args.fps,
        width=args.width, height=args.height,
    ))
    tl.sources = [source]
    tl.speakers = dict(SPEAKER_NAMES)
    tl.save(ws.edl)
    print(f"ingested {source.duration:.1f}s\n  proxy: {proxy}\n  audio: {audio}")
    return tl


def stage_transcribe(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    src = next(s for s in tl.sources if s.role == "main")
    audio = ingest.extract_audio(src.path, ws.work / "audio")
    if args.remote and not remote.is_configured():
        raise SystemExit(
            "--remote given but RunPod is not configured; see the README, "
            "or drop --remote to transcribe locally")
    if args.remote:
        print("  transcribing on RunPod (large-v3)")
    subs, review, meta = transcribe.run(
        audio, speaker_names=tl.speakers, model=args.model, device=args.device,
        remote=args.remote,
    )
    tl.subtitles = subs
    tl.review.extend(review)
    tl.save(ws.edl)
    ws.srt.write_text(transcribe.to_srt(subs), encoding="utf-8")
    ws.transcript.write_text(
        transcribe.to_prompt_transcript(subs), encoding="utf-8")
    print(f"transcribed {len(subs)} segments (diarized={meta['diarized']})")
    print(f"  srt: {ws.srt}")
    if not meta["diarized"]:
        print("  note: no speaker labels — install pyannote.audio and set HF_TOKEN "
              "for name cards and per-speaker punch-ins")
    return tl


def stage_analyse(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    if not tl.subtitles:
        raise SystemExit("no transcript yet — run the transcribe stage first")
    tl.cuts, tl.overlays, tl.chapters, tl.reframes = [], [], [], []
    tl.review = [r for r in tl.review if r.kind == "diarization"]
    analyse.run(tl, tl.subtitles, client=_openrouter_client())

    problems = tl.validate()
    if problems:
        raise SystemExit("analysis produced an invalid timeline:\n  "
                         + "\n  ".join(problems))
    tl.conform()
    tl.save(ws.edl)
    print(f"analysed: {len(tl.cuts)} cuts, {len(tl.overlays)} overlays, "
          f"{len(tl.chapters)} chapters, {len(tl.reframes)} reframes")
    print(f"  runtime {tl.duration / 60:.1f} min "
          f"(from {tl.sources[0].duration / 60:.1f} min raw)")
    print(f"  {len(tl.blocking_review())} item(s) need sign-off before render")
    return tl


def stage_graphics(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    from . import brand
    if not brand.has_brand_font():
        print("warning: Fredoka is not installed — cards will render in a "
              "fallback font and will be off-brand")
    graphics.run(tl, ws.graphics)
    tl.save(ws.edl)
    print(f"rendered {len(tl.overlays)} cards into {ws.graphics}")
    return tl


def stage_assemble(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    if not tl.conformed:
        raise SystemExit("timeline is not conformed — re-run the analyse stage")
    missing = [o.kind for o in tl.overlays if not o.asset]
    if missing:
        raise SystemExit(f"{len(missing)} overlay(s) have no rendered card — "
                         f"run the graphics stage first")
    path = kdenlive.run(tl, ws.project)
    chapters = analyse.youtube_chapters(tl)
    (ws.dir / "chapters.txt").write_text(chapters, encoding="utf-8")
    print(f"wrote {path}")
    print(f"  chapters: {ws.dir / 'chapters.txt'}")
    return tl


def stage_thumbnail(ws: Workspace, args) -> Timeline:
    """
    Two-part stage. Without --pick it proposes a shortlist and stops; with
    --pick it builds the finished thumbnail from the frame you chose.

    The pause is deliberate. Frame scoring rejects blinks and blur reliably,
    but it ranks neutral expressions highest, and neutral makes a dull
    thumbnail. The machine narrows; you choose.
    """
    tl = ws.load()
    src = next(s for s in tl.sources if s.role == "main")
    template = args.template or (ASSETS / "templates" / "thumbnail.png")

    if args.pick is None and args.frame is None:
        windows = None
        if args.at is not None:
            windows = [(max(0.0, args.at - 2.0), args.at + 2.0)]
        cands = thumbnail_stage.propose(
            tl, src.path, ws.thumb_work, top=args.top, windows=windows,
            remote=args.remote)
        if not cands:
            raise SystemExit(
                "no usable frames found — every candidate was a blink, a "
                "mid-word mouth or motion blur. Try --at <seconds>.")
        print(f"{len(cands)} candidate(s), best first:\n")
        for i, c in enumerate(cands):
            print(f"  [{i:>2}] {c.label():>6}  score {c.score:.2f}  "
                  f"eye {c.eye:.2f} mouth {c.mouth:.2f} sharp {c.sharpness:.0f}")
        print(f"\n  contact sheet: {ws.thumb_work / 'contact-sheet.png'}")
        print(f"\nPick one with: run.py thumbnail --slug {ws.slug} --pick <n>")
        return tl

    if args.frame:
        frame = Path(args.frame)
    else:
        cands = json.loads((ws.thumb_work / "candidates.json").read_text())
        if args.pick >= len(cands):
            raise SystemExit(f"--pick {args.pick} out of range (0..{len(cands) - 1})")
        frame = Path(cands[args.pick]["path"])

    phrase = args.phrase or _phrase_from_metadata(ws)
    out = thumbnail_stage.build(frame, ws.thumb_work, ws.thumbnail,
                                phrase=phrase, template=template,
                                remote=args.remote)
    print(f"thumbnail: {out}")
    if not phrase:
        print("  no phrase set — run the metadata stage first, or pass --phrase")
    return tl


def _phrase_from_metadata(ws: Workspace) -> str:
    """Reuse the thumbnail phrase the metadata stage generated, when it exists."""
    cache = ws.work / "metadata.json"
    if cache.exists():
        return json.loads(cache.read_text()).get("thumbnail_phrase", "")
    return ""


def stage_metadata(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    if not tl.subtitles:
        raise SystemExit("no transcript yet — run the transcribe stage first")

    # Regenerating replaces the previous copy's review items rather than
    # stacking a second set of them on the timeline.
    tl.review = [r for r in tl.review
                 if r.kind not in ("metadata_claim", "metadata_copy")]

    transcript = transcribe.to_prompt_transcript(tl.subtitles)
    meta = metadata_stage.generate(tl, transcript, client=_openrouter_client())
    (ws.work / "metadata.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
    chapters = analyse.youtube_chapters(tl)
    out = metadata_stage.render_file(tl, meta, chapters, ws.metadata)
    tl.save(ws.edl)

    print(f"metadata: {out}")
    print(f"  title: {meta['title']}")
    print(f"  phrase: {meta['thumbnail_phrase']}")
    print(f"  {len(tl.blocking_review())} item(s) need sign-off before render")
    return tl


def stage_review(ws: Workspace, args) -> Timeline:
    """Print the human checkpoint, and optionally clear items."""
    tl = ws.load()
    if args.resolve:
        wanted = set(args.resolve)
        for i, item in enumerate(tl.review):
            if str(i) in wanted or "all" in wanted:
                item.resolved = True
        tl.save(ws.edl)

    if not tl.review:
        print("nothing flagged for review")
        return tl

    print(f"{len(tl.review)} review item(s):\n")
    for i, item in enumerate(tl.review):
        mark = "done" if item.resolved else item.severity.upper()
        m, s = divmod(int(item.at), 60)
        print(f"  [{i:>2}] {mark:<5} {m:>2}:{s:02d}  {item.kind}: {item.note}")
    open_blocking = len(tl.blocking_review())
    print(f"\n{open_blocking} blocking item(s) still open.")
    if open_blocking:
        print("Resolve with: run.py review --slug %s --resolve 0 1 2" % ws.slug)
    return tl


def stage_render(ws: Workspace, args) -> Timeline:
    tl = ws.load()
    if not ws.project.exists():
        raise SystemExit("no project file — run the assemble stage first")
    out = render_stage.run(tl, ws.project, ws.output,
                           skip_review_gate=args.force)
    print(f"rendered {out}")
    return tl


STAGES = {
    "ingest": stage_ingest,
    "transcribe": stage_transcribe,
    "analyse": stage_analyse,
    "graphics": stage_graphics,
    "thumbnail": stage_thumbnail,
    "metadata": stage_metadata,
    "assemble": stage_assemble,
    "review": stage_review,
    "render": stage_render,
}

# "all" deliberately stops before thumbnail: that stage needs you to pick a
# frame. Everything up to a finished project file and publishing copy is
# automatic; the two human choices (the frame, the review gate) are not.
ALL_ORDER = ["ingest", "transcribe", "analyse", "graphics", "metadata",
             "assemble", "review"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--slug", required=True, help="episode id, e.g. 2026-09-06")
    p.add_argument("--root", default=DEFAULT_ROOT, type=Path)
    p.add_argument("--raw", help="path to the camera file (ingest only)")
    p.add_argument("--title", default="")
    p.add_argument("--recorded", default="")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--model", default="base", help="whisper model size")
    p.add_argument("--device", default="cpu")
    p.add_argument("--resolve", nargs="*", default=None,
                   help="review item indices to mark resolved, or 'all'")
    p.add_argument("--force", action="store_true",
                   help="render despite open blocking review items")
    p.add_argument("--top", type=int, default=10,
                   help="thumbnail: how many candidate frames to shortlist")
    p.add_argument("--pick", type=int, default=None,
                   help="thumbnail: build from candidate N of the shortlist")
    p.add_argument("--frame", default=None,
                   help="thumbnail: build from this image, skipping selection")
    p.add_argument("--at", type=float, default=None,
                   help="thumbnail: search near this raw timestamp, in seconds")
    p.add_argument("--phrase", default=None,
                   help="thumbnail: text block override")
    p.add_argument("--template", type=Path, default=None,
                   help="thumbnail: background template PNG")
    p.add_argument("--remote", action="store_true",
                   help="run the GPU-heavy work on RunPod (transcribe, thumbnail)")
    args = p.parse_args(argv)

    ws = Workspace(args.slug, args.root)

    if args.stage == "all":
        if not args.raw:
            p.error("--raw is required for 'all'")
        for name in ALL_ORDER:
            print(f"\n=== {name} ===")
            STAGES[name](ws, args)
        print("\n=== next ===")
        print(f"  1. pick a thumbnail:  run.py thumbnail --slug {args.slug}")
        print(f"  2. clear the review:  run.py review --slug {args.slug}")
        print(f"  3. then, if you want a file: run.py render --slug {args.slug}")
        print(f"\n  project:   {ws.project}")
        print(f"  metadata:  {ws.metadata}")
        return 0

    if args.stage == "ingest" and not args.raw:
        p.error("--raw is required for 'ingest'")

    STAGES[args.stage](ws, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
