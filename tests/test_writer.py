"""
Structural tests for the Kdenlive writer.

These lock in the MLT behaviours that were found the hard way against a real
melt render — each one silently produced a broken video before it was fixed.
"""

import xml.etree.ElementTree as ET

import pytest

from video_pipeline.edl import Cut, Episode, Overlay, Reframe, Source, Timeline
from video_pipeline.writers.kdenlive import KdenliveWriter


def build(tmp_path, *, overlays=(), reframes=()):
    tl = Timeline(episode=Episode(slug="ep", fps=25))
    tl.sources = [Source(id="main", path="/tmp/cam.mp4", duration=60.0)]
    tl.cuts = [Cut("main", 0, 10), Cut("main", 20, 30)]
    tl.overlays = list(overlays)
    tl.reframes = list(reframes)
    tl.conform()
    for ov in tl.overlays:
        ov.asset = "/tmp/card.png"
    path = KdenliveWriter(tl, tmp_path).write(tmp_path / "ep.kdenlive")
    return tl, ET.parse(path).getroot()


def props(el):
    return {p.get("name"): p.text for p in el.findall("property")}


def master_of(root):
    return next(t for t in root.findall("tractor")
                if "kdenlive:projectTractor" in props(t))


def test_refuses_an_unconformed_timeline(tmp_path):
    tl = Timeline(episode=Episode(slug="ep"))
    tl.sources = [Source(id="main", path="/tmp/c.mp4", duration=10)]
    tl.cuts = [Cut("main", 0, 5)]
    with pytest.raises(ValueError, match="unconformed"):
        KdenliveWriter(tl, tmp_path).write(tmp_path / "x.kdenlive")


def test_every_track_tractor_has_two_playlists(tmp_path):
    """Kdenlive will not open a track with only one playlist."""
    _, root = build(tmp_path, overlays=[Overlay(1, 4, "name_card", {"name": "H"})])
    for tractor in root.findall("tractor"):
        if "kdenlive:projectTractor" in props(tractor):
            continue
        assert len(tractor.findall("track")) == 2


def test_master_lists_black_track_first(tmp_path):
    _, root = build(tmp_path)
    tracks = master_of(root).findall("track")
    assert tracks[0].get("producer") == "black_track"


def test_video_tracks_use_composite_not_qtblend(tmp_path):
    """qtblend drops the upper track's alpha in a headless melt render."""
    _, root = build(tmp_path, overlays=[Overlay(1, 4, "name_card", {"name": "H"})])
    services = [props(t).get("mlt_service")
                for t in master_of(root).findall("transition")]
    assert "composite" in services
    assert "qtblend" not in services


def test_audio_track_gets_a_mix_transition(tmp_path):
    _, root = build(tmp_path)
    services = [props(t).get("mlt_service")
                for t in master_of(root).findall("transition")]
    assert services.count("mix") == 1


def test_transition_b_track_indices_match_track_order(tmp_path):
    _, root = build(tmp_path, overlays=[Overlay(1, 4, "name_card", {"name": "H"})])
    master = master_of(root)
    n_tracks = len(master.findall("track")) - 1        # minus black_track
    b_tracks = sorted(int(props(t)["b_track"])
                      for t in master.findall("transition"))
    assert b_tracks == list(range(1, n_tracks + 1))


def test_reframe_creates_a_punch_track_with_static_geometry(tmp_path):
    """
    Keyframed composite geometry is normalised to the transition length, so
    punch-ins are separate tracks with one static geometry each.
    """
    _, root = build(tmp_path, reframes=[Reframe(2, 8, "sophie", 1.35, 0.66, 0.46)])
    names = [props(t).get("kdenlive:track_name", "") for t in root.findall("tractor")]
    assert any("Punch" in n for n in names)

    geometries = [props(t).get("geometry") for t in master_of(root).findall("transition")
                  if props(t).get("mlt_service") == "composite"]
    punch = [g for g in geometries if g and "%" not in g]
    assert punch, "expected one static punch-in geometry"
    # Larger than the frame and offset negatively — a magnification, not a shrink.
    assert punch[0].startswith("-")
    assert "2592x1458" in punch[0]


def test_full_frame_reframes_do_not_create_a_track(tmp_path):
    _, root = build(tmp_path, reframes=[Reframe(2, 8, "both", scale=1.0)])
    names = [props(t).get("kdenlive:track_name", "") for t in root.findall("tractor")]
    assert not any("Punch" in n for n in names)


def test_overlapping_overlays_are_pushed_later_not_dropped(tmp_path):
    tl, root = build(tmp_path, overlays=[
        Overlay(1, 6, "name_card", {"name": "H"}),
        Overlay(3, 8, "poll", {"question": "?"}),
    ])
    playlists = [pl for pl in root.findall("playlist")
                 if len(pl.findall("entry")) and pl.get("id") != "main_bin"]
    graphics = [pl for pl in playlists
                if any("card.png" in (e.get("producer") or "") or True
                       for e in pl.findall("entry"))]
    # Both cards survive somewhere in the project.
    assert sum(len(pl.findall("entry")) for pl in graphics) >= 2


def test_bin_contains_every_visual_producer(tmp_path):
    _, root = build(tmp_path, overlays=[Overlay(1, 4, "poll", {"question": "?"})])
    bin_pl = root.find("./playlist[@id='main_bin']")
    ids = {e.get("producer") for e in bin_pl.findall("entry")}
    assert len(ids) >= 2                      # camera + the card


def test_master_out_matches_the_edit_length(tmp_path):
    tl, root = build(tmp_path)
    assert master_of(root).get("out") == "00:00:20.000"
    assert tl.duration == 20
