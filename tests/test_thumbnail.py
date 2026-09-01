"""
The halo maths, without the models.

Frame scoring needs mediapipe and a real face, and the cutout needs rembg;
neither belongs in a unit test. What is worth locking down is the image
geometry, because it is the part that silently produces a *rounded* outline
instead of a sticker one and still looks like it worked.
"""

from pathlib import Path

import pytest

from video_pipeline.stages import thumbnail

PIL = pytest.importorskip("PIL", reason="Pillow is needed for the halo")
from PIL import Image, ImageDraw  # noqa: E402


def _square_cutout(tmp_path: Path, soft: bool = False) -> Path:
    """A hard-edged square on transparency — corners make rounding visible."""
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 149, 149], fill=(255, 0, 0, 255))
    if soft:
        from PIL import ImageFilter
        alpha = img.getchannel("A").filter(ImageFilter.GaussianBlur(4))
        img.putalpha(alpha)
    path = tmp_path / ("soft.png" if soft else "hard.png")
    img.save(path)
    return path


def test_binary_alpha_has_no_partial_transparency():
    """The whole sticker look depends on this: no half-opaque edge pixels."""
    img = Image.new("RGBA", (60, 60), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([10, 10, 49, 49], fill=(255, 255, 255, 255))
    from PIL import ImageFilter
    img.putalpha(img.getchannel("A").filter(ImageFilter.GaussianBlur(3)))

    assert any(0 < v < 255 for v in img.getchannel("A").tobytes())
    hardened = thumbnail._binary_alpha(img, erode=0)
    assert set(hardened.getchannel("A").tobytes()) <= {0, 255}


def test_dilate_preserves_corners():
    """
    A dilated square must stay square.

    This is the difference between the Legal Eagle outline and a soft glow: if
    the corner pixel is empty after growing, the structuring element rounded it
    and the outline will read as a blob.
    """
    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle([30, 30, 69, 69], fill=255)

    grown = thumbnail._dilate(mask, 10)

    assert grown.getpixel((20, 20)) == 255, "corner rounded off during dilate"
    assert grown.getpixel((79, 79)) == 255
    assert grown.getpixel((30, 30)) == 255
    # ...but it must not grow further than asked.
    assert grown.getpixel((18, 18)) == 0


def test_dilate_width_is_exact_beyond_the_filter_cap():
    """Radii over 5 are applied in passes; the total must still be exact."""
    mask = Image.new("L", (200, 200), 0)
    ImageDraw.Draw(mask).rectangle([80, 80, 119, 119], fill=255)

    grown = thumbnail._dilate(mask, 22)

    assert grown.getpixel((80 - 22, 100)) == 255
    assert grown.getpixel((80 - 23, 100)) == 0


def test_halo_output_is_larger_and_still_transparent(tmp_path):
    cut = _square_cutout(tmp_path)
    out = thumbnail.halo(cut, tmp_path / "sticker.png")
    img = Image.open(out).convert("RGBA")

    assert img.width > 200 and img.height > 200, "canvas must grow for the stroke"
    assert img.getpixel((0, 0))[3] == 0, "outside the sticker must stay transparent"

    # The subject survives unmodified in the middle.
    centre = img.getpixel((img.width // 2, img.height // 2))
    assert centre[:3] == (255, 0, 0) and centre[3] == 255


def test_halo_draws_strokes_in_width_order(tmp_path):
    """
    Widest first, so the narrow stroke sits on top.

    Given a 26px dark and a 14px white stroke, a pixel 20px out from the
    subject is in the dark band, and one 8px out is white. Reversed ordering
    would paint the dark over the white and lose the white edge entirely.
    """
    cut = _square_cutout(tmp_path)
    out = thumbnail.halo(
        cut, tmp_path / "s.png",
        strokes=[{"width": 26, "colour": "#000000"},
                 {"width": 14, "colour": "#FFFFFF"}],
        shadow={"offset": (0, 0), "blur": 0, "opacity": 0, "grow": 0},
    )
    img = Image.open(out).convert("RGBA")

    # Subject's left edge sat at x=50 in a 200px image, offset by the pad.
    pad = (img.width - 200) // 2
    left_edge = pad + 50
    y = img.height // 2

    assert img.getpixel((left_edge - 8, y))[:3] == (255, 255, 255), "white band"
    assert img.getpixel((left_edge - 20, y))[:3] == (0, 0, 0), "dark keyline"
    assert img.getpixel((left_edge - 30, y))[3] == 0, "nothing beyond the widest"


def _edge_width(img, y: int) -> int:
    """How many pixels the alpha takes to go from clear to solid on one row."""
    row = [img.getpixel((x, y))[3] for x in range(img.width)]
    partial = [i for i, a in enumerate(row) if 8 < a < 247]
    if not partial:
        return 0
    # Only the leading edge; the trailing one is a separate transition.
    first = partial[0]
    width = 1
    while first + width in partial:
        width += 1
    return width


def test_soft_cutout_still_yields_a_die_cut_edge(tmp_path):
    """
    The regression that motivated all of this.

    A feathered cutout — what rembg actually returns — must still produce a
    crisp outline, because halo() thresholds before it grows. The edge is
    antialiased by the supersampled downscale, so it is not strictly binary,
    but the transition must be a pixel or two rather than the original feather.
    Remove the threshold and this blows out to the blur radius.
    """
    cut = _square_cutout(tmp_path, soft=True)
    out = thumbnail.halo(
        cut, tmp_path / "s.png",
        strokes=[{"width": 12, "colour": "#FFFFFF"}],
        shadow={"offset": (0, 0), "blur": 0, "opacity": 0, "grow": 0},
    )
    img = Image.open(out).convert("RGBA")
    assert _edge_width(img, img.height // 2) <= 3, "stroke edge is feathered"


def test_supersample_one_is_exactly_binary(tmp_path):
    """The underlying geometry is hard; antialiasing is only the downscale."""
    cut = _square_cutout(tmp_path, soft=True)
    out = thumbnail.halo(
        cut, tmp_path / "s1.png",
        strokes=[{"width": 12, "colour": "#FFFFFF"}],
        shadow={"offset": (0, 0), "blur": 0, "opacity": 0, "grow": 0},
        supersample=1,
    )
    alphas = set(Image.open(out).getchannel("A").tobytes())
    assert alphas <= {0, 255}


def test_merge_windows_combines_overlaps():
    merged = thumbnail._merge([(0.0, 4.0), (3.0, 6.0), (10.0, 12.0)])
    assert merged == [(0.0, 6.0), (10.0, 12.0)]


def test_spread_rejects_near_duplicate_frames():
    """
    Ten frames from one well-lit second is not a choice.

    Candidates arrive sorted by score; _spread must skip anything within
    min_gap of an already-picked frame even when its score is higher than
    later, more distant frames.
    """
    def cand(t, score):
        return thumbnail.Candidate(path="", raw_time=t, timeline_time=t,
                                   score=score, eye=0.3, mouth=0.2,
                                   sharpness=100.0, yaw=0.1)

    scored = [cand(10.0, 0.9), cand(10.2, 0.88), cand(10.4, 0.87),
              cand(30.0, 0.5), cand(60.0, 0.4)]
    picked = thumbnail._spread(scored, top=3, min_gap=1.5)

    assert [c.raw_time for c in picked] == [10.0, 30.0, 60.0]


@pytest.mark.parametrize("measured,expected_reason", [
    ({"eye": 0.05, "mouth": 0.2, "sharpness": 200, "yaw": 0.1, "face_h": 0.2},
     "eyes closed / mid-blink"),
    ({"eye": 0.3, "mouth": 0.9, "sharpness": 200, "yaw": 0.1, "face_h": 0.2},
     "mouth open mid-word"),
    ({"eye": 0.3, "mouth": 0.2, "sharpness": 5, "yaw": 0.1, "face_h": 0.2},
     "motion blur"),
])
def test_disqualifying_faults_score_zero(measured, expected_reason):
    score, reason = thumbnail._combine(measured)
    assert score == 0.0
    assert reason == expected_reason


def test_a_good_frame_scores_above_zero():
    score, reason = thumbnail._combine(
        {"eye": 0.30, "mouth": 0.25, "sharpness": 400, "yaw": 0.05, "face_h": 0.2})
    assert reason == ""
    assert 0.9 < score <= 1.0
