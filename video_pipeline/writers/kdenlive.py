"""
kdenlive.py — write a conformed Timeline out as a .kdenlive (MLT XML) project.

Structure, matching what Kdenlive itself produces:

    <mlt>
      <profile/>
      <producer>            one per distinct clip/graphic/colour
      <playlist id=main_bin>  the project bin
      playlist pair + tractor   per timeline track
      <tractor>             master: black_track + every track + transitions

Each Kdenlive track is a *tractor* wrapping two playlists — Kdenlive uses the
second for same-track transitions. We only ever fill the first, but both must
exist or the project will not open.

Track order in the master tractor is audio first (bottom of the UI), then video.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from ..edl import Timeline

KDENLIVE_VERSION = "24.05.2"
MLT_VERSION = "7.24.0"


def tc(seconds: float, fps: int) -> str:
    """MLT timecode. Rounded to the frame grid so clips butt up exactly."""
    frames = max(0, round(seconds * fps))
    total_ms = int(round(frames / fps * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _prop(parent: ET.Element, name: str, value) -> ET.Element:
    el = ET.SubElement(parent, "property", {"name": name})
    el.text = str(value)
    return el


class KdenliveWriter:
    def __init__(self, timeline: Timeline, project_dir: str | Path):
        if not timeline.conformed:
            raise ValueError(
                "refusing to write an unconformed timeline — call Timeline.conform() "
                "first, or overlays will land at raw-footage times"
            )
        self.tl = timeline
        self.fps = timeline.episode.fps
        self.dir = Path(project_dir)
        self.root = ET.Element("mlt", {
            "LC_NUMERIC": "C", "version": MLT_VERSION,
            "producer": "main_bin", "root": str(self.dir.resolve()),
        })
        self._n = 0
        self._producers: dict[tuple, str] = {}
        self._bin: list[str] = []
        self._geometry: dict[str, str] = {}

    # ---- ids ---------------------------------------------------------------

    def _next(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    # ---- profile -----------------------------------------------------------

    def _profile(self):
        ep = self.tl.episode
        ET.SubElement(self.root, "profile", {
            "description": f"{ep.width}x{ep.height} {ep.fps}fps",
            "width": str(ep.width), "height": str(ep.height),
            "progressive": "1", "colorspace": "709",
            "sample_aspect_num": "1", "sample_aspect_den": "1",
            "display_aspect_num": "16", "display_aspect_den": "9",
            "frame_rate_num": str(ep.fps), "frame_rate_den": "1",
        })

    # ---- producers ---------------------------------------------------------

    def _av_producer(self, path: str, duration: float, kind: str) -> str:
        """
        One producer per (file, kind). Kdenlive needs separate producers for the
        video and audio halves of the same file when they sit on different
        tracks, hence `kind` in the cache key.
        """
        key = (path, kind)
        if key in self._producers:
            return self._producers[key]

        pid = self._next("producer")
        length = max(1, round(duration * self.fps))
        p = ET.SubElement(self.root, "producer", {
            "id": pid, "in": tc(0, self.fps),
            "out": tc(duration, self.fps),
        })
        _prop(p, "length", length)
        _prop(p, "eof", "pause")
        _prop(p, "resource", path)
        _prop(p, "mlt_service", "avformat")
        _prop(p, "seekable", "1")
        _prop(p, "audio_index", "-1" if kind == "video" else "1")
        _prop(p, "video_index", "-1" if kind == "audio" else "0")
        _prop(p, "kdenlive:id", str(self._n))
        _prop(p, "kdenlive:clipname", Path(path).name)
        self._producers[key] = pid
        if kind != "audio":
            self._bin.append(pid)
        return pid

    def _image_producer(self, path: str, duration: float) -> str:
        key = (path, "image")
        if key in self._producers:
            return self._producers[key]
        pid = self._next("producer")
        p = ET.SubElement(self.root, "producer", {
            "id": pid, "in": tc(0, self.fps), "out": tc(duration, self.fps),
        })
        # A generous length lets the same card be reused at any duration.
        _prop(p, "length", max(1, round(duration * self.fps)) + self.fps * 60)
        _prop(p, "eof", "continue")
        _prop(p, "resource", path)
        _prop(p, "mlt_service", "qimage")
        _prop(p, "ttl", "1")
        _prop(p, "kdenlive:id", str(self._n))
        _prop(p, "kdenlive:clipname", Path(path).name)
        self._producers[key] = pid
        self._bin.append(pid)
        return pid

    def _black_track(self):
        p = ET.SubElement(self.root, "producer", {"id": "black_track"})
        _prop(p, "length", "2147483647")
        _prop(p, "eof", "continue")
        _prop(p, "resource", "black")
        _prop(p, "aspect_ratio", "1")
        _prop(p, "mlt_service", "color")
        _prop(p, "mlt_image_format", "rgba")
        _prop(p, "set.test_audio", "0")

    # ---- bin ---------------------------------------------------------------

    def _main_bin(self):
        pl = ET.SubElement(self.root, "playlist", {"id": "main_bin"})
        docprops = {
            "kdenliveversion": KDENLIVE_VERSION,
            "profile": f"{self.tl.episode.width}x{self.tl.episode.height}",
            "version": "1.1",
            "audioChannels": "2",
            "documentid": self.tl.episode.slug,
            "enableproxy": "0",
            "renderurl": f"Output/{self.tl.episode.slug}.mp4",
        }
        for k, v in docprops.items():
            _prop(pl, f"kdenlive:docproperties.{k}", v)
        _prop(pl, "kdenlive:documentnotes",
              f"Generated by video-pipeline from {self.tl.episode.slug}.edl.json. "
              f"Re-run the assemble stage rather than hand-editing this header.")
        _prop(pl, "xml_retain", "1")
        for pid in self._bin:
            ET.SubElement(pl, "entry", {"producer": pid})

    # ---- tracks ------------------------------------------------------------

    def _track(self, name: str, entries: list, is_audio: bool) -> str:
        """
        Build one timeline track: two playlists inside a tractor.

        `entries` is a list of ("blank", seconds) or ("clip", pid, in, out) tuples,
        already in timeline order.
        """
        pl_a, pl_b = self._next("playlist"), self._next("playlist")
        for pid in (pl_a, pl_b):
            pl = ET.SubElement(self.root, "playlist", {"id": pid})
            if is_audio:
                _prop(pl, "kdenlive:audio_track", "1")

        playlist = self.root.find(f"./playlist[@id='{pl_a}']")
        for item in entries:
            if item[0] == "blank":
                ET.SubElement(playlist, "blank", {"length": tc(item[1], self.fps)})
            else:
                _, producer, tin, tout = item
                ET.SubElement(playlist, "entry", {
                    "producer": producer,
                    "in": tc(tin, self.fps), "out": tc(tout, self.fps),
                })

        tid = self._next("tractor")
        tractor = ET.SubElement(self.root, "tractor", {"id": tid})
        if is_audio:
            _prop(tractor, "kdenlive:audio_track", "1")
        _prop(tractor, "kdenlive:track_name", name)
        _prop(tractor, "kdenlive:trackheight", "70")
        _prop(tractor, "kdenlive:timeline_active", "1")
        _prop(tractor, "kdenlive:collapsed", "0")
        hide = "video" if is_audio else "audio"
        for pid in (pl_a, pl_b):
            ET.SubElement(tractor, "track", {"hide": hide, "producer": pid})
        return tid

    # ---- track content -----------------------------------------------------

    def _main_entries(self, kind: str) -> list:
        """The main camera cuts, as timeline entries."""
        src = next(s for s in self.tl.sources if s.role == "main")
        pid = self._av_producer(src.path, src.duration, kind)
        return [("clip", pid, c.start, c.end) for c in self.tl.cuts]

    def _overlay_entries(self) -> list:
        """Graphics track: cards separated by blanks."""
        entries, cursor = [], 0.0
        for ov in sorted(self.tl.overlays, key=lambda o: o.start):
            if not ov.asset:
                continue
            if ov.start < cursor:
                # Overlapping cards would need a second graphics track; push
                # this one later rather than silently dropping it.
                ov.start, ov.end = cursor, cursor + (ov.end - ov.start)
            if ov.start > cursor:
                entries.append(("blank", ov.start - cursor))
            duration = ov.end - ov.start
            pid = self._image_producer(ov.asset, duration)
            entries.append(("clip", pid, 0.0, duration))
            cursor = ov.end
        return entries

    def _sfx_entries(self) -> list:
        entries, cursor = [], 0.0
        by_name = {s.id: s for s in self.tl.sources if s.role == "sfx"}
        for cue in sorted(self.tl.sfx, key=lambda s: s.start):
            src = by_name.get(cue.name)
            if src is None:
                continue
            if cue.start > cursor:
                entries.append(("blank", cue.start - cursor))
            pid = self._av_producer(src.path, src.duration, "audio")
            entries.append(("clip", pid, 0.0, src.duration))
            cursor = max(cursor, cue.start + src.duration)
        return entries

    # ---- reframe keyframes -------------------------------------------------

    def _punch_groups(self) -> list[tuple[tuple, list]]:
        """
        Group reframes by the framing they ask for.

        Each distinct (scale, x, y) becomes one timeline track holding copies of
        the camera cut for those spans, with a single STATIC composite geometry.
        In practice that is one track per host, which is also how a human would
        cut it.

        Static geometry is used deliberately: MLT's `composite` normalises
        keyframed geometry to the transition's own length, so multi-keyframe
        animation smears across the whole episode instead of landing on the
        span it was written for. Verified against melt.
        """
        groups: dict[tuple, list] = {}
        for rf in sorted(self.tl.reframes, key=lambda r: r.start):
            if rf.scale == 1.0:
                continue
            groups.setdefault((rf.scale, rf.x, rf.y), []).append(rf)
        return list(groups.items())

    def _punch_geometry(self, scale: float, x: float, y: float) -> str:
        """Destination rect for a punch-in: larger than frame, offset negatively."""
        ep = self.tl.episode
        w, h = ep.width * scale, ep.height * scale
        return f"{-(w - ep.width) * x:.0f},{-(h - ep.height) * y:.0f}:{w:.0f}x{h:.0f}:100"

    def _punch_entries(self, reframes: list) -> list:
        """Camera clips for one punch group, as timeline entries."""
        entries, cursor = [], 0.0
        for rf in reframes:
            origin = self.tl.timeline_to_raw(rf.start)
            if origin is None:
                continue
            _, raw_start = origin
            if rf.start > cursor:
                entries.append(("blank", rf.start - cursor))
            src = next(s for s in self.tl.sources if s.role == "main")
            pid = self._av_producer(src.path, src.duration, "video")
            entries.append(("clip", pid, raw_start, raw_start + (rf.end - rf.start)))
            cursor = rf.end
        return entries

    # ---- master ------------------------------------------------------------

    def _master(self, audio_tracks: list[str], video_tracks: list[str]):
        master = ET.SubElement(self.root, "tractor", {
            "id": self._next("tractor"),
            "in": tc(0, self.fps), "out": tc(self.tl.duration, self.fps),
        })
        _prop(master, "kdenlive:projectTractor", "1")
        ET.SubElement(master, "track", {"producer": "black_track"})

        # Audio below video, matching Kdenlive's own ordering.
        ordered = list(reversed(audio_tracks)) + video_tracks
        for tid in ordered:
            ET.SubElement(master, "track", {"producer": tid})

        for i, tid in enumerate(ordered, start=1):
            is_audio = tid in audio_tracks
            tr = ET.SubElement(master, "transition", {"id": self._next("transition")})
            _prop(tr, "a_track", "0")
            _prop(tr, "b_track", str(i))
            _prop(tr, "always_active", "1")
            _prop(tr, "internal_added", "237")
            if is_audio:
                _prop(tr, "mlt_service", "mix")
                _prop(tr, "kdenlive_id", "mix")
                _prop(tr, "accepts_blanks", "1")
                _prop(tr, "sum", "1")
            else:
                # `composite`, not `qtblend`. qtblend discards the upper
                # track's alpha when rendered headlessly through melt, so a
                # transparent graphics card punches straight through to black
                # and hides the camera beneath it. Verified against both.
                _prop(tr, "mlt_service", "composite")
                _prop(tr, "geometry", self._geometry.get(tid, "0=0,0:100%x100%:100"))
                _prop(tr, "aligned", "0")
                _prop(tr, "distort", "0")
                _prop(tr, "fill", "1")

    # ---- build -------------------------------------------------------------

    def build(self) -> ET.Element:
        self._profile()
        self._black_track()

        # Producers must exist before the bin is written, so build content first.
        video_entries = self._main_entries("video")
        audio_entries = self._main_entries("audio")
        overlay_entries = self._overlay_entries()
        punch_groups = self._punch_groups()
        sfx_entries = self._sfx_entries()

        self._main_bin()

        a1 = self._track("A1 Dialogue", audio_entries, is_audio=True)
        audio_tracks = [a1]
        if sfx_entries:
            audio_tracks.append(self._track("A2 SFX", sfx_entries, is_audio=True))

        v1 = self._track("V1 Camera", video_entries, is_audio=False)
        video_tracks = [v1]

        # One track per distinct punch-in framing, above the base camera.
        for i, ((scale, x, y), reframes) in enumerate(punch_groups, start=1):
            entries = self._punch_entries(reframes)
            if not entries:
                continue
            label = reframes[0].target.title()
            tid = self._track(f"V{i + 1} Punch — {label}", entries, is_audio=False)
            self._geometry[tid] = self._punch_geometry(scale, x, y)
            video_tracks.append(tid)

        if overlay_entries:
            video_tracks.append(
                self._track(f"V{len(video_tracks) + 1} Graphics",
                            overlay_entries, is_audio=False))

        self._master(audio_tracks, video_tracks)
        return self.root

    def write(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.build()
        xml = ET.tostring(self.root, encoding="unicode")
        pretty = minidom.parseString(xml).toprettyxml(indent=" ")
        # minidom pads text nodes; collapse the damage on property values.
        pretty = "\n".join(ln for ln in pretty.splitlines() if ln.strip())
        out_path.write_text(pretty, encoding="utf-8")
        return out_path


def run(timeline: Timeline, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    return KdenliveWriter(timeline, out_path.parent).write(out_path)
