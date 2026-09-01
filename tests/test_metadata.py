"""
Metadata generation, with the model stubbed.

The value here is not the copy — that is the model's job. It is that generated
copy cannot reach a published surface without passing the review gate, and that
the Bluesky draft never carries a live link.
"""

import json

import pytest

from video_pipeline import llm
from video_pipeline.edl import Episode, Subtitle, Timeline
from video_pipeline.stages import metadata


@pytest.fixture
def timeline():
    tl = Timeline(episode=Episode(slug="2026-09-06", recorded="2026-09-06"))
    tl.subtitles = [Subtitle(start=0.0, end=4.0, text="Hello.", speaker="Heather")]
    return tl


@pytest.fixture
def fake_model(monkeypatch):
    """Stub the LLM chain; tests assert on handling, not on generation."""
    def install(payload):
        monkeypatch.setattr(metadata.llm, "classify_json",
                            lambda prompt, client=None: payload)
    return install


BASE = {
    "title": "Something Happened In 2026",
    "description": "Keywords first. Then the body.",
    "thumbnail_phrase": "IT'S THE LAW",
    "bluesky": "New video on the ruling. It affects more people than you think.",
    "tags": ["politics", "uk"],
    "claims": [],
}


def test_generated_copy_blocks_the_render(timeline, fake_model):
    """Title and social copy are channel voice; they do not auto-publish."""
    fake_model(BASE)
    metadata.generate(timeline, "transcript")

    blocking = [r for r in timeline.blocking_review()]
    kinds = {r.kind for r in blocking}
    assert "metadata_copy" in kinds
    assert any("title:" in r.note for r in blocking)
    assert any("bluesky:" in r.note for r in blocking)


def test_factual_claims_are_flagged_individually(timeline, fake_model):
    fake_model({**BASE, "claims": [
        {"claim": "Support fell 12 points since 2024", "where": "description"},
        {"claim": "Three MPs voted against", "where": "bluesky"},
    ]})
    metadata.generate(timeline, "transcript")

    claims = [r for r in timeline.review if r.kind == "metadata_claim"]
    assert len(claims) == 2
    assert all(r.severity == "block" for r in claims)


def test_a_url_from_the_model_is_stripped(timeline, fake_model):
    """
    The model is told not to include a link and sometimes does anyway.

    A live URL in the draft is how an unreviewed post gets published by
    reflex, so it is removed rather than trusted.
    """
    fake_model({**BASE,
                "bluesky": "Watch it here https://youtu.be/abc123 — worth your time."})
    meta = metadata.generate(timeline, "transcript")

    assert "http" not in meta["bluesky"]
    assert "youtu.be" not in meta["bluesky"]


def test_overlong_title_is_flagged_but_not_discarded(timeline, fake_model):
    long_title = "A" * 80
    fake_model({**BASE, "title": long_title})
    meta = metadata.generate(timeline, "transcript")

    assert meta["title"] == long_title, "the copy is kept for you to trim"
    assert any("over the 60 limit" in r.note for r in timeline.review)


def test_regenerating_replaces_rather_than_stacks(timeline, fake_model):
    """Re-running the stage must not leave two generations of review items."""
    fake_model(BASE)
    metadata.generate(timeline, "transcript")
    first = len([r for r in timeline.review if r.kind.startswith("metadata")])

    timeline.review = [r for r in timeline.review
                       if r.kind not in ("metadata_claim", "metadata_copy")]
    metadata.generate(timeline, "transcript")
    second = len([r for r in timeline.review if r.kind.startswith("metadata")])

    assert first == second


def test_file_contains_the_link_placeholder(timeline, fake_model, tmp_path):
    fake_model(BASE)
    meta = metadata.generate(timeline, "transcript")
    out = metadata.render_file(timeline, meta, "0:00 Intro", tmp_path / "m.txt")
    text = out.read_text()

    assert metadata.LINK_PLACEHOLDER in text
    assert "not posted" in text
    assert meta["title"] in text
    assert "0:00 Intro" in text


def test_file_lists_open_review_items(timeline, fake_model, tmp_path):
    """
    The text file is what gets opened at upload time, so the warning has to be
    in it — not only in the terminal output of a command run an hour earlier.
    """
    fake_model({**BASE, "claims": [{"claim": "Unverified thing", "where": "description"}]})
    meta = metadata.generate(timeline, "transcript")
    out = metadata.render_file(timeline, meta, "", tmp_path / "m.txt")

    text = out.read_text()
    assert "UNRESOLVED REVIEW ITEMS" in text
    assert "Unverified thing" in text


def test_env_file_does_not_override_real_environment(tmp_path, monkeypatch):
    """A systemd unit or CI variable must win over the checked-out .env."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=from-file\nOTHER=from-file\n")
    monkeypatch.setenv("VIDEO_TOOLS_ENV", str(env_file))
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-environment")
    llm.load_env.cache_clear()

    llm.load_env()

    assert llm.env("OPENROUTER_API_KEY") == "from-environment"
    assert llm.env("OTHER") == "from-file"
    llm.load_env.cache_clear()
