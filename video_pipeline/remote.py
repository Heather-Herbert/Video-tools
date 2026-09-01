"""
remote.py — run the GPU-heavy stages on RunPod instead of this machine.

The desktop this runs on is short of graphics power, and two stages care:
transcription (Whisper) and the thumbnail's background removal. Everything
else — analysis, card rendering, XML assembly — is cheap and stays local.

Transport
---------
RunPod serverless passes JSON, and its payload limits are far too small for
audio or video. So media goes through S3-compatible object storage and the job
payload carries only keys:

    local ──upload audio──> S3 ──key──> RunPod worker ──result JSON──> local

Any S3-compatible store works: Cloudflare R2, Backblaze B2, AWS, or a RunPod
network volume with the S3 gateway enabled. The bucket only ever holds
intermediates, so a lifecycle rule that deletes objects after a day is enough,
and `cleanup()` removes them anyway on success.

Failure policy
--------------
Remote is an optimisation, never a requirement. Every entry point here falls
back to local execution when RunPod is not configured, and `--remote` is opt-in
per run. A cold GPU worker takes 30-60s to boot, so a short clip is often
faster locally; the win is on a full episode.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path

from . import llm

POLL_INTERVAL = 5.0
# A 45-minute episode on a cold worker: boot, model load, then transcription.
DEFAULT_TIMEOUT = 3600.0


class RemoteError(RuntimeError):
    """The remote run failed. The caller may fall back to local."""


class RemoteNotConfigured(RemoteError):
    """No endpoint or credentials. Expected on a fresh checkout, not a bug."""


# --- configuration ----------------------------------------------------------

def config() -> dict:
    """Read remote settings from .env. Missing values are empty strings."""
    llm.load_env()
    return {
        "api_key": llm.env("RUNPOD_API_KEY"),
        "endpoint": llm.env("RUNPOD_ENDPOINT_ID"),
        "bucket": llm.env("S3_BUCKET"),
        "s3_endpoint": llm.env("S3_ENDPOINT_URL"),
        "s3_key": llm.env("S3_ACCESS_KEY_ID"),
        "s3_secret": llm.env("S3_SECRET_ACCESS_KEY"),
        "region": llm.env("S3_REGION", "auto"),
    }


def is_configured() -> bool:
    cfg = config()
    return bool(cfg["api_key"] and cfg["endpoint"]
                and cfg["bucket"] and cfg["s3_key"])


def _require_config() -> dict:
    cfg = config()
    missing = [k for k in ("api_key", "endpoint", "bucket", "s3_key", "s3_secret")
               if not cfg[k]]
    if missing:
        raise RemoteNotConfigured(
            "RunPod is not configured; missing " + ", ".join(sorted(missing))
            + ". See the 'Running stages on RunPod' section of the README, "
              "or drop --remote to run locally."
        )
    return cfg


# --- object storage ---------------------------------------------------------

def _s3(cfg: dict):
    try:
        import boto3  # noqa: PLC0415
    except ImportError as exc:
        raise RemoteError("boto3 is needed for --remote; pip install boto3") from exc
    return boto3.client(
        "s3",
        endpoint_url=cfg["s3_endpoint"] or None,
        aws_access_key_id=cfg["s3_key"],
        aws_secret_access_key=cfg["s3_secret"],
        region_name=cfg["region"],
    )


def upload(path: Path, cfg: dict, prefix: str = "in") -> str:
    """Put a file in the bucket under a unique key and return the key."""
    key = f"{prefix}/{uuid.uuid4().hex}/{Path(path).name}"
    _s3(cfg).upload_file(str(path), cfg["bucket"], key)
    return key


def download(key: str, dest: Path, cfg: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _s3(cfg).download_file(cfg["bucket"], key, str(dest))
    return dest


def cleanup(keys: list[str], cfg: dict) -> None:
    """Best-effort removal of intermediates. Never fatal — they expire anyway."""
    if not keys:
        return
    try:
        _s3(cfg).delete_objects(
            Bucket=cfg["bucket"],
            Delete={"Objects": [{"Key": k} for k in keys]},
        )
    except Exception:                                  # noqa: BLE001
        pass


# --- job control ------------------------------------------------------------

def _post(url: str, cfg: dict, payload: dict | None = None) -> dict:
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Authorization": f"Bearer {cfg['api_key']}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise RemoteError(
            f"RunPod returned {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RemoteError(f"cannot reach RunPod: {exc.reason}") from exc


def submit(payload: dict, cfg: dict) -> str:
    """Queue a job asynchronously and return its id."""
    base = f"https://api.runpod.ai/v2/{cfg['endpoint']}"
    result = _post(f"{base}/run", cfg, {"input": payload})
    job_id = result.get("id")
    if not job_id:
        raise RemoteError(f"RunPod did not return a job id: {result}")
    return job_id


def wait(job_id: str, cfg: dict, timeout: float = DEFAULT_TIMEOUT,
         on_status=None) -> dict:
    """
    Poll until the job finishes.

    Long jobs are submitted async rather than sync because a synchronous
    request times out at the HTTP layer well before a 45-minute transcript
    finishes, and the job then runs on invisibly, billed and unreachable.
    """
    base = f"https://api.runpod.ai/v2/{cfg['endpoint']}"
    deadline = time.time() + timeout
    last = None

    while time.time() < deadline:
        state = _post(f"{base}/status/{job_id}", cfg)
        status = state.get("status")
        if status != last:
            last = status
            if on_status:
                on_status(status)
        if status == "COMPLETED":
            return state.get("output") or {}
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RemoteError(
                f"remote job {status}: {state.get('error', 'no error given')}")
        time.sleep(POLL_INTERVAL)

    # Do not leave a job running and billing after we have stopped caring.
    try:
        _post(f"{base}/cancel/{job_id}", cfg)
    except RemoteError:
        pass
    raise RemoteError(f"remote job timed out after {timeout:.0f}s (cancelled)")


def run_job(payload: dict, cfg: dict, timeout: float = DEFAULT_TIMEOUT,
            verbose: bool = True) -> dict:
    job_id = submit(payload, cfg)
    if verbose:
        print(f"  runpod job {job_id} queued")
    return wait(job_id, cfg, timeout,
                on_status=(lambda s: print(f"  {s.lower()}")) if verbose else None)


# --- the stages -------------------------------------------------------------

def transcribe(audio_path: Path, model: str = "large-v3",
               language: str = "en", diarize: bool = False,
               verbose: bool = True) -> dict:
    """
    Transcribe on a GPU worker. Returns the same dict shape as the local path.

    The model defaults to large-v3 rather than the local default: the point of
    renting a GPU is to stop trading accuracy for a CPU's patience.
    """
    cfg = _require_config()
    if verbose:
        print(f"  uploading {audio_path.name} ({audio_path.stat().st_size / 1e6:.1f} MB)")
    key = upload(audio_path, cfg, prefix="audio")
    try:
        output = run_job({
            "stage": "transcribe",
            "audio_key": key,
            "model": model,
            "language": language,
            "diarize": diarize,
            "bucket": cfg["bucket"],
        }, cfg, verbose=verbose)
    finally:
        cleanup([key], cfg)

    if "segments" not in output:
        raise RemoteError(f"worker returned no segments: {str(output)[:300]}")
    return output


def cutout(frame_path: Path, out_path: Path, verbose: bool = True) -> Path:
    """
    Remove a background on a GPU worker.

    A single frame is small enough to pass inline as base64, which avoids a
    bucket round-trip for what is usually under a megabyte.
    """
    cfg = _require_config()
    encoded = base64.b64encode(frame_path.read_bytes()).decode()
    output = run_job({"stage": "cutout", "image_b64": encoded},
                     cfg, timeout=600.0, verbose=verbose)

    if not output.get("image_b64"):
        raise RemoteError(f"worker returned no image: {str(output)[:300]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(output["image_b64"]))
    return out_path


def score_frames(frame_paths: list[Path], verbose: bool = True) -> list[dict]:
    """
    Score a batch of frames remotely.

    Batched in one job because per-frame jobs would pay the worker's cold-start
    cost dozens of times over, which is slower than doing it locally.
    """
    cfg = _require_config()
    payload = {
        "stage": "score",
        "frames": [
            {"name": p.name,
             "image_b64": base64.b64encode(p.read_bytes()).decode()}
            for p in frame_paths
        ],
    }
    output = run_job(payload, cfg, timeout=1800.0, verbose=verbose)
    return output.get("scores", [])
