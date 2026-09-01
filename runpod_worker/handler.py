"""
handler.py — the RunPod serverless worker.

Runs the two stages worth renting a GPU for: Whisper transcription and the
thumbnail's background removal plus face scoring. Everything else in the
pipeline is cheap and stays on the desktop.

This file runs *inside the container*, not on the desktop, and deliberately
imports nothing from `video_pipeline`. The worker's only contract is the JSON
shape it returns, so the image can be rebuilt without the pipeline and the
pipeline can be edited without rebuilding the image.

Models are loaded once into module globals and reused. RunPod keeps a warm
worker alive between jobs, so a second job on the same worker skips the model
load entirely — which is most of the wall time on short jobs.

Input:  {"stage": "transcribe", "audio_key": ..., "bucket": ..., "model": ...}
        {"stage": "cutout",     "image_b64": ...}
        {"stage": "score",      "frames": [{"name": ..., "image_b64": ...}]}
Output: {"segments": [...]} | {"image_b64": ...} | {"scores": [...]}
        {"error": "..."} on failure.
"""

import base64
import io
import os
import tempfile
import traceback

import runpod

_whisper = None
_whisper_key = None
_rembg_session = None
_face_mesh = None


# --- lazily loaded models ---------------------------------------------------

def get_whisper(model: str):
    """Load (and cache) the Whisper model. Reused across jobs on a warm worker."""
    global _whisper, _whisper_key
    if _whisper is None or _whisper_key != model:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(model, device="cuda", compute_type="float16")
        _whisper_key = model
    return _whisper


def get_rembg():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )
    return _face_mesh


def s3():
    import boto3
    from botocore.config import Config

    # See remote.py: boto3 1.36+ sends CRC32 checksums by default, which R2
    # and B2 reject. Harmless to disable on AWS.
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("S3_REGION", "auto"),
        config=Config(request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"),
    )


# --- stages -----------------------------------------------------------------

def do_transcribe(job_input: dict) -> dict:
    bucket = job_input.get("bucket") or os.environ["S3_BUCKET"]
    key = job_input["audio_key"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    s3().download_file(bucket, key, audio_path)

    try:
        model = get_whisper(job_input.get("model", "large-v3"))
        segments, info = model.transcribe(
            audio_path, language=job_input.get("language", "en"),
            vad_filter=True, word_timestamps=True,
        )
        # faster-whisper is lazy: transcription happens as this is consumed.
        out = [
            {
                "start": seg.start, "end": seg.end, "text": seg.text.strip(),
                "words": [{"start": w.start, "end": w.end, "word": w.word}
                          for w in (seg.words or [])],
            }
            for seg in segments
        ]
        result = {"segments": out, "language": info.language,
                  "text": " ".join(s["text"] for s in out)}

        if job_input.get("diarize"):
            result["speakers"] = do_diarize(audio_path)
        return result
    finally:
        os.unlink(audio_path)


def do_diarize(audio_path: str) -> list:
    """Speaker turns, when a HF token is baked into the worker's environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return []
    try:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token)
        import torch
        pipeline.to(torch.device("cuda"))
        annotation = pipeline(audio_path, num_speakers=2)
        return [
            {"start": turn.start, "end": turn.end, "speaker": label,
             "confidence": 1.0}
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]
    except Exception:
        # Diarization is optional; the pipeline flags unlabelled segments for a
        # human instead. Losing the whole transcript over it would be worse.
        traceback.print_exc()
        return []


def do_cutout(job_input: dict) -> dict:
    from PIL import Image
    from rembg import remove

    raw = base64.b64decode(job_input["image_b64"])
    img = Image.open(io.BytesIO(raw)).convert("RGBA")
    result = remove(img, session=get_rembg())

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return {"image_b64": base64.b64encode(buf.getvalue()).decode()}


def do_score(job_input: dict) -> dict:
    """
    Score frames for blinks, mouth shape, sharpness and pose.

    The measurement geometry is duplicated from stages/thumbnail.py rather than
    imported, because this container does not ship the pipeline. The thresholds
    deliberately are *not* duplicated: this returns raw measurements and the
    desktop decides what disqualifies a frame, so tuning does not need a
    container rebuild.
    """
    import math

    import numpy as np
    from PIL import Image

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    MOUTH = (61, 81, 311, 291, 402, 178)
    NOSE, CHIN, L_CHEEK, R_CHEEK = 1, 152, 234, 454

    def aspect(pts):
        d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])  # noqa: E731
        width = d(pts[0], pts[3])
        if width < 1e-6:
            return 0.0
        return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * width)

    mesh = get_face_mesh()
    scores = []
    for frame in job_input.get("frames", []):
        raw = base64.b64decode(frame["image_b64"])
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        result = mesh.process(arr)
        if not result.multi_face_landmarks:
            scores.append({"name": frame["name"], "face": False})
            continue

        h, w = arr.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        pt = lambda i: (lm[i].x * w, lm[i].y * h)  # noqa: E731

        eye = (aspect([pt(i) for i in LEFT_EYE])
               + aspect([pt(i) for i in RIGHT_EYE])) / 2.0
        mouth = aspect([pt(i) for i in MOUTH])
        lc, rc, nose = pt(L_CHEEK), pt(R_CHEEK), pt(NOSE)
        span = abs(rc[0] - lc[0]) or 1.0
        yaw = abs(((nose[0] - lc[0]) / span) - 0.5) * 2.0

        xs = [p[0] for p in (nose, pt(CHIN), lc, rc)]
        ys = [p[1] for p in (nose, pt(CHIN), lc, rc)]
        x0, x1 = max(0, int(min(xs)) - 40), min(w, int(max(xs)) + 40)
        y0, y1 = max(0, int(min(ys)) - 80), min(h, int(max(ys)) + 40)
        crop = arr[y0:y1, x0:x1]
        grey = crop.mean(axis=2) if crop.size else np.zeros((2, 2))
        if grey.size > 16:
            lap = (grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2]
                   + grey[1:-1, 2:] - 4 * grey[1:-1, 1:-1])
        else:
            lap = np.zeros((1,))

        scores.append({
            "name": frame["name"], "face": True,
            "eye": float(eye), "mouth": float(mouth), "yaw": float(yaw),
            "sharpness": float(lap.var()),
            "face_h": float(abs(pt(CHIN)[1] - nose[1]) / h),
        })

    return {"scores": scores}


STAGES = {"transcribe": do_transcribe, "cutout": do_cutout, "score": do_score}


def handler(job):
    job_input = job.get("input") or {}
    stage = job_input.get("stage")
    fn = STAGES.get(stage)
    if fn is None:
        return {"error": f"unknown stage {stage!r}; expected one of "
                         f"{sorted(STAGES)}"}
    try:
        return fn(job_input)
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
