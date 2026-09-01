"""
edl.py — the tool-neutral edit decision list.

This is the contract between the analysis half of the pipeline (transcribe,
LLM analysis, graphics) and the assembly half (Kdenlive today, Resolve/FCPXML
later). Nothing upstream of this module knows what editor we use; nothing
downstream re-runs an LLM.

Two clocks matter and confusing them is the main way this pipeline goes wrong:

    raw time       seconds into the original camera file
    timeline time  seconds into the finished edit, after cuts are removed

The LLM analyses a transcript, so everything it emits is in *raw* time.
Overlays and SFX have to land on the *timeline*. `Timeline.conform` does that
translation once, at a single point, so no other stage has to think about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

SCHEMA = "polycule-edl/1"

# Overlay kinds the graphics renderer and the Kdenlive writer both understand.
OVERLAY_KINDS = (
    "name_card",
    "citation",
    "pull_quote",
    "chapter",
    "stat",
    "reaction",
    "poll",
    "further_reading",
    "correction",
    "end_teaser",
)

# Overlay kinds a human must sign off before render. Citations and corrections
# make factual claims on screen in the channel's name; they never auto-publish.
NEEDS_REVIEW = ("citation", "stat", "correction", "further_reading")


@dataclass
class Source:
    """A media file the timeline draws on."""
    id: str
    path: str
    role: str = "main"          # main | stinger | sfx | graphic | music
    duration: float = 0.0
    has_video: bool = True
    has_audio: bool = True


@dataclass
class Cut:
    """A kept segment of a source, in raw source time."""
    src: str
    start: float
    end: float
    reason: str = ""            # why the gap before this cut was removed

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Reframe:
    """A punch-in toward one speaker. Raw time in, timeline time after conform."""
    start: float
    end: float
    target: str = "both"        # heather | sophie | both
    scale: float = 1.0          # 1.0 = full frame, 1.4 = 40% punch in
    x: float = 0.5              # normalised centre of the crop
    y: float = 0.5


@dataclass
class Overlay:
    """A graphic card. `fields` carries the per-episode text for the template."""
    start: float
    end: float
    kind: str
    fields: dict = field(default_factory=dict)
    asset: str | None = None    # filled in by the graphics stage
    track: str = "V3"

    def __post_init__(self):
        if self.kind not in OVERLAY_KINDS:
            raise ValueError(
                f"unknown overlay kind {self.kind!r}; expected one of {OVERLAY_KINDS}"
            )


@dataclass
class Sfx:
    start: float
    name: str
    gain_db: float = 0.0


@dataclass
class Chapter:
    start: float
    title: str


@dataclass
class Subtitle:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class ReviewItem:
    """Something a human must look at before this episode renders."""
    kind: str                   # citation | ambiguous_cut | diarization | stat
    at: float
    note: str
    severity: str = "check"     # check | block
    resolved: bool = False


@dataclass
class Episode:
    slug: str
    title: str = ""
    recorded: str = ""
    fps: int = 25
    width: int = 1920
    height: int = 1080


@dataclass
class Timeline:
    episode: Episode
    sources: list[Source] = field(default_factory=list)
    speakers: dict = field(default_factory=dict)
    cuts: list[Cut] = field(default_factory=list)
    reframes: list[Reframe] = field(default_factory=list)
    overlays: list[Overlay] = field(default_factory=list)
    sfx: list[Sfx] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    subtitles: list[Subtitle] = field(default_factory=list)
    review: list[ReviewItem] = field(default_factory=list)
    conformed: bool = False

    # ---- time mapping -----------------------------------------------------

    def raw_to_timeline(self, t: float) -> float | None:
        """
        Map a raw source time to its position in the edited timeline.

        Returns None if `t` falls in material that was cut out — the caller
        decides whether to drop the event or snap it. Cuts are assumed sorted
        and non-overlapping, which `validate` enforces.
        """
        elapsed = 0.0
        for c in self.cuts:
            if t < c.start:
                return None                    # inside a removed gap
            if t <= c.end:
                return elapsed + (t - c.start)
            elapsed += c.duration
        return None                            # past the end of the last cut

    def timeline_to_raw(self, t: float) -> tuple[str, float] | None:
        """
        Inverse of raw_to_timeline: which source and source-time does this
        timeline position come from? Needed to place a punch-in clip, which
        must reference the original footage at the right offset.
        """
        elapsed = 0.0
        for c in self.cuts:
            if t < elapsed + c.duration:
                return c.src, c.start + (t - elapsed)
            elapsed += c.duration
        return None

    def _snap(self, t: float) -> float:
        """Map raw→timeline, snapping into the nearest kept segment if cut."""
        mapped = self.raw_to_timeline(t)
        if mapped is not None:
            return mapped
        # Fall to the start of the next kept segment, or the end of the edit.
        elapsed = 0.0
        for c in self.cuts:
            if t < c.start:
                return elapsed
            elapsed += c.duration
        return elapsed

    def conform(self) -> "Timeline":
        """
        Rewrite every raw-time event onto the edited timeline. Idempotent-guarded:
        conforming twice would silently compress everything, so it refuses.
        """
        if self.conformed:
            raise RuntimeError("timeline already conformed; conform() is not idempotent")
        if not self.cuts:
            raise RuntimeError("cannot conform with no cuts — run the analysis stage first")

        for group in (self.reframes, self.overlays):
            for ev in group:
                ev.start, ev.end = self._snap(ev.start), self._snap(ev.end)
        for s in self.sfx:
            s.start = self._snap(s.start)
        for ch in self.chapters:
            ch.start = self._snap(ch.start)
        for sub in self.subtitles:
            sub.start, sub.end = self._snap(sub.start), self._snap(sub.end)
        for r in self.review:
            r.at = self._snap(r.at)

        # A zero-length event survived a cut that swallowed it whole. Drop it
        # rather than emit a degenerate clip the editor will choke on.
        self.overlays = [o for o in self.overlays if o.end - o.start > 0.04]
        self.reframes = [r for r in self.reframes if r.end - r.start > 0.04]
        self.subtitles = [s for s in self.subtitles if s.end - s.start > 0.04]

        self.conformed = True
        return self

    @property
    def duration(self) -> float:
        return sum(c.duration for c in self.cuts)

    # ---- integrity --------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of problems. Empty list means the timeline is sane."""
        problems = []
        ids = {s.id for s in self.sources}
        if len(ids) != len(self.sources):
            problems.append("duplicate source ids")

        prev = None
        for i, c in enumerate(self.cuts):
            if c.src not in ids:
                problems.append(f"cut {i} references unknown source {c.src!r}")
            if c.end <= c.start:
                problems.append(f"cut {i} has non-positive duration")
            if prev is not None and c.start < prev:
                problems.append(f"cut {i} starts before the previous cut ends (overlap)")
            prev = c.end

        for i, o in enumerate(self.overlays):
            if o.end <= o.start:
                problems.append(f"overlay {i} ({o.kind}) has non-positive duration")

        for r in self.reframes:
            if not 0.5 <= r.scale <= 3.0:
                problems.append(f"reframe scale {r.scale} outside sane range 0.5–3.0")

        return problems

    def blocking_review(self) -> list[ReviewItem]:
        """Review items that must be resolved before render is allowed."""
        return [r for r in self.review if r.severity == "block" and not r.resolved]

    # ---- persistence ------------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({"schema": SCHEMA, **asdict(self)}, indent=indent)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Timeline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = data.pop("schema", None)
        if schema != SCHEMA:
            raise ValueError(f"expected schema {SCHEMA}, got {schema!r}")
        return cls(
            episode=Episode(**data["episode"]),
            sources=[Source(**s) for s in data.get("sources", [])],
            speakers=data.get("speakers", {}),
            cuts=[Cut(**c) for c in data.get("cuts", [])],
            reframes=[Reframe(**r) for r in data.get("reframes", [])],
            overlays=[Overlay(**o) for o in data.get("overlays", [])],
            sfx=[Sfx(**s) for s in data.get("sfx", [])],
            chapters=[Chapter(**c) for c in data.get("chapters", [])],
            subtitles=[Subtitle(**s) for s in data.get("subtitles", [])],
            review=[ReviewItem(**r) for r in data.get("review", [])],
            conformed=data.get("conformed", False),
        )
