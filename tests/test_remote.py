"""
The remote layer, with RunPod stubbed.

No network here. What matters is the job lifecycle: a failed job must raise
rather than return junk, a job we stop waiting on must be cancelled rather than
left running and billing, and missing configuration must say so plainly instead
of failing somewhere deep in boto3.
"""

import pytest

from video_pipeline import remote


@pytest.fixture
def cfg():
    return {"api_key": "k", "endpoint": "e", "bucket": "b",
            "s3_endpoint": "https://s3.example", "s3_key": "id",
            "s3_secret": "secret", "region": "auto"}


@pytest.fixture
def calls(monkeypatch):
    """Record every RunPod HTTP call and reply from a scripted queue."""
    log = []

    def install(responses):
        queue = list(responses)

        def fake_post(url, cfg, payload=None):
            log.append(url)
            return queue.pop(0) if queue else {"status": "IN_QUEUE"}

        monkeypatch.setattr(remote, "_post", fake_post)
        monkeypatch.setattr(remote.time, "sleep", lambda _: None)
        return log

    return install


def test_missing_config_names_what_is_missing(monkeypatch):
    monkeypatch.setattr(remote, "config", lambda: {
        "api_key": "", "endpoint": "", "bucket": "", "s3_key": "",
        "s3_secret": "", "s3_endpoint": "", "region": "auto"})

    with pytest.raises(remote.RemoteNotConfigured) as exc:
        remote._require_config()

    message = str(exc.value)
    assert "api_key" in message and "bucket" in message
    assert "--remote" in message, "should say how to proceed without RunPod"


def test_is_configured_is_false_when_incomplete(monkeypatch):
    monkeypatch.setattr(remote, "config", lambda: {
        "api_key": "k", "endpoint": "e", "bucket": "", "s3_key": "id",
        "s3_secret": "s", "s3_endpoint": "", "region": "auto"})
    assert remote.is_configured() is False


def test_wait_returns_output_on_completion(cfg, calls):
    calls([{"status": "IN_QUEUE"},
           {"status": "IN_PROGRESS"},
           {"status": "COMPLETED", "output": {"segments": [{"text": "hi"}]}}])

    output = remote.wait("job-1", cfg)

    assert output == {"segments": [{"text": "hi"}]}


def test_failed_job_raises_with_the_error(cfg, calls):
    calls([{"status": "FAILED", "error": "CUDA out of memory"}])

    with pytest.raises(remote.RemoteError, match="CUDA out of memory"):
        remote.wait("job-1", cfg)


def test_timeout_cancels_the_job(cfg, calls):
    """
    A job we have stopped waiting for must be cancelled.

    Otherwise it keeps running on a rented GPU, billing, with nothing left to
    collect the result.
    """
    log = calls([{"status": "IN_PROGRESS"}] * 50)

    with pytest.raises(remote.RemoteError, match="timed out"):
        remote.wait("job-1", cfg, timeout=0.0)

    assert any("/cancel/job-1" in url for url in log), "job left running"


def test_submit_rejects_a_reply_with_no_job_id(cfg, calls):
    calls([{"error": "endpoint not found"}])
    with pytest.raises(remote.RemoteError, match="did not return a job id"):
        remote.submit({"stage": "transcribe"}, cfg)


def test_transcribe_uploads_then_cleans_up(cfg, monkeypatch, tmp_path):
    """Intermediates must not accumulate in the bucket after a successful run."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 1024)
    deleted = []

    monkeypatch.setattr(remote, "_require_config", lambda: cfg)
    monkeypatch.setattr(remote, "upload", lambda p, c, prefix="in": "audio/k/a.wav")
    monkeypatch.setattr(remote, "cleanup", lambda keys, c: deleted.extend(keys))
    monkeypatch.setattr(remote, "run_job",
                        lambda payload, c, **kw: {"segments": [], "text": ""})

    remote.transcribe(audio, verbose=False)

    assert deleted == ["audio/k/a.wav"]


def test_transcribe_cleans_up_even_when_the_job_fails(cfg, monkeypatch, tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    deleted = []

    def boom(payload, c, **kw):
        raise remote.RemoteError("worker died")

    monkeypatch.setattr(remote, "_require_config", lambda: cfg)
    monkeypatch.setattr(remote, "upload", lambda p, c, prefix="in": "audio/k/a.wav")
    monkeypatch.setattr(remote, "cleanup", lambda keys, c: deleted.extend(keys))
    monkeypatch.setattr(remote, "run_job", boom)

    with pytest.raises(remote.RemoteError):
        remote.transcribe(audio, verbose=False)

    assert deleted == ["audio/k/a.wav"], "upload orphaned in the bucket"


def test_transcribe_rejects_a_reply_with_no_segments(cfg, monkeypatch, tmp_path):
    """A worker error dict must not be mistaken for an empty transcript."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(remote, "_require_config", lambda: cfg)
    monkeypatch.setattr(remote, "upload", lambda p, c, prefix="in": "k")
    monkeypatch.setattr(remote, "cleanup", lambda keys, c: None)
    monkeypatch.setattr(remote, "run_job",
                        lambda payload, c, **kw: {"error": "CUDA OOM"})

    with pytest.raises(remote.RemoteError, match="no segments"):
        remote.transcribe(audio, verbose=False)


def test_cutout_round_trips_base64(cfg, monkeypatch, tmp_path):
    import base64

    src = tmp_path / "frame.png"
    src.write_bytes(b"PNGDATA")
    sent = {}

    def fake_run_job(payload, c, **kw):
        sent.update(payload)
        return {"image_b64": base64.b64encode(b"CUTOUT").decode()}

    monkeypatch.setattr(remote, "_require_config", lambda: cfg)
    monkeypatch.setattr(remote, "run_job", fake_run_job)

    out = remote.cutout(src, tmp_path / "out.png", verbose=False)

    assert base64.b64decode(sent["image_b64"]) == b"PNGDATA"
    assert out.read_bytes() == b"CUTOUT"


def test_score_frames_batches_into_one_job(cfg, monkeypatch, tmp_path):
    """
    All frames go in a single job.

    One job per frame would pay the worker's cold-start cost dozens of times,
    which is slower than never leaving the desktop.
    """
    frames = []
    for i in range(4):
        f = tmp_path / f"f{i}.png"
        f.write_bytes(b"x")
        frames.append(f)

    jobs = []
    monkeypatch.setattr(remote, "_require_config", lambda: cfg)
    monkeypatch.setattr(remote, "run_job",
                        lambda payload, c, **kw: jobs.append(payload) or {"scores": []})

    remote.score_frames(frames, verbose=False)

    assert len(jobs) == 1
    assert len(jobs[0]["frames"]) == 4
