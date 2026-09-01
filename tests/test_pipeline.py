"""Tests for the parts of the pipeline that are easy to get subtly wrong."""

import pytest

from video_pipeline.edl import (
    Chapter, Cut, Episode, Overlay, Reframe, ReviewItem, Source, Subtitle, Timeline,
)
from video_pipeline.stages.analyse import build_cuts, youtube_chapters
from video_pipeline.writers.kdenlive import tc


def make_timeline(cuts):
    tl = Timeline(episode=Episode(slug="t", fps=25))
    tl.sources = [Source(id="main", path="/tmp/x.mp4", duration=100.0)]
    tl.cuts = [Cut("main", a, b) for a, b in cuts]
    return tl


# --- raw <-> timeline mapping ----------------------------------------------

def test_raw_to_timeline_inside_kept_segments():
    tl = make_timeline([(0, 10), (20, 30)])
    assert tl.raw_to_timeline(0) == 0
    assert tl.raw_to_timeline(5) == 5
    assert tl.raw_to_timeline(20) == 10       # start of the second kept segment
    assert tl.raw_to_timeline(25) == 15


def test_raw_to_timeline_returns_none_in_a_removed_gap():
    tl = make_timeline([(0, 10), (20, 30)])
    assert tl.raw_to_timeline(15) is None
    assert tl.raw_to_timeline(99) is None


def test_snap_moves_a_cut_event_to_the_next_kept_segment():
    tl = make_timeline([(0, 10), (20, 30)])
    assert tl._snap(15) == 10


def test_timeline_to_raw_is_the_inverse():
    tl = make_timeline([(0, 10), (20, 30)])
    for raw in (0.0, 5.0, 9.9, 20.0, 25.0):
        mapped = tl.raw_to_timeline(raw)
        assert tl.timeline_to_raw(mapped) == ("main", pytest.approx(raw))


def test_duration_excludes_removed_gaps():
    assert make_timeline([(0, 10), (20, 30)]).duration == 20


# --- conform ----------------------------------------------------------------

def test_conform_moves_overlays_onto_the_edited_timeline():
    tl = make_timeline([(0, 10), (20, 30)])
    tl.overlays = [Overlay(22, 26, "reaction", {"text": "hi"})]
    tl.chapters = [Chapter(20, "Second topic")]
    tl.conform()
    assert (tl.overlays[0].start, tl.overlays[0].end) == (12, 16)
    assert tl.chapters[0].start == 10


def test_conform_drops_events_swallowed_by_a_cut():
    tl = make_timeline([(0, 10), (20, 30)])
    tl.overlays = [Overlay(12, 13, "reaction", {"text": "gone"})]
    tl.conform()
    assert tl.overlays == []


def test_conform_refuses_to_run_twice():
    tl = make_timeline([(0, 10)])
    tl.conform()
    with pytest.raises(RuntimeError, match="already conformed"):
        tl.conform()


def test_conform_refuses_without_cuts():
    tl = make_timeline([])
    with pytest.raises(RuntimeError, match="no cuts"):
        tl.conform()


# --- cut building -----------------------------------------------------------

def subs(*spans):
    return [Subtitle(a, b, "text", "Heather") for a, b in spans]


def test_short_gaps_are_merged_not_cut():
    cuts = build_cuts(subs((0, 5), (5.3, 9)), "main", 20, [], [])
    assert len(cuts) == 1                      # 0.3s gap is below MIN_GAP


def test_long_gaps_are_cut():
    cuts = build_cuts(subs((0, 5), (12, 18)), "main", 30, [], [])
    assert len(cuts) == 2


def test_a_protected_beat_is_kept_but_capped():
    cuts = build_cuts(subs((0, 5), (12, 18)), "main", 30, protected=[11.8], drops=[])
    # The beat is preserved as trailing air, capped at MAX_PROTECTED_BEAT.
    assert cuts[0].end == pytest.approx(5.18 + 1.6)


def test_dropped_spans_are_removed_entirely():
    cuts = build_cuts(subs((0, 5), (12, 18), (25, 30)), "main", 40,
                      [], drops=[(11, 19)])
    assert len(cuts) == 2
    assert all(not (11 < c.start < 19) for c in cuts)


# --- validation and the review gate -----------------------------------------

def test_validate_catches_overlapping_cuts():
    tl = make_timeline([(0, 10), (5, 15)])
    assert any("overlap" in p for p in tl.validate())


def test_validate_catches_unknown_source():
    tl = make_timeline([(0, 10)])
    tl.cuts.append(Cut("nope", 20, 30))
    assert any("unknown source" in p for p in tl.validate())


def test_blocking_review_only_counts_unresolved_blockers():
    tl = make_timeline([(0, 10)])
    tl.review = [
        ReviewItem("citation", 1, "verify", "block"),
        ReviewItem("citation", 2, "verified", "block", resolved=True),
        ReviewItem("ambiguous_cut", 3, "look", "check"),
    ]
    assert len(tl.blocking_review()) == 1


def test_unknown_overlay_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown overlay kind"):
        Overlay(0, 1, "banana", {})


def test_reframe_scale_out_of_range_is_flagged():
    tl = make_timeline([(0, 10)])
    tl.reframes = [Reframe(0, 5, "heather", scale=9.0)]
    assert any("scale" in p for p in tl.validate())


# --- round trip and formatting ----------------------------------------------

def test_timeline_survives_a_save_load_round_trip(tmp_path):
    tl = make_timeline([(0, 10), (20, 30)])
    tl.overlays = [Overlay(1, 4, "name_card", {"name": "Sophie"})]
    tl.review = [ReviewItem("citation", 2, "check it", "block")]
    tl.conform()
    path = tl.save(tmp_path / "t.edl.json")
    back = Timeline.load(path)
    assert back.conformed
    assert back.duration == tl.duration
    assert back.overlays[0].fields["name"] == "Sophie"
    assert len(back.blocking_review()) == 1


def test_timecode_snaps_to_the_frame_grid():
    assert tc(0, 25) == "00:00:00.000"
    assert tc(1.0, 25) == "00:00:01.000"
    assert tc(3.041, 25) == "00:00:03.040"     # rounded to the nearest frame
    assert tc(-5, 25) == "00:00:00.000"        # never negative


def test_youtube_chapters_always_start_at_zero():
    tl = make_timeline([(0, 100)])
    tl.chapters = [Chapter(30, "Second thing")]
    out = youtube_chapters(tl)
    assert out.splitlines()[0] == "0:00 Intro"
    assert "0:30 Second thing" in out
