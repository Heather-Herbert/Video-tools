# GPU worker for the transcribe / cutout / score stages.
#
# Build and push (from the repo root):
#   docker build -t <dockerhub-user>/video-tools-worker:1 runpod_worker/
#   docker push  <dockerhub-user>/video-tools-worker:1
#
# Then point a RunPod serverless endpoint at that image. See the README.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Bake the rembg model into the image. Downloading it on first request would
# add ~40s to a cold start and needs egress from the worker.
RUN python3 -c "from rembg import new_session; new_session('u2net')"

# Pre-download Whisper too, for the same reason. Change this if you change the
# default model in remote.transcribe().
ARG WHISPER_MODEL=large-v3
RUN python3 -c "from faster_whisper import WhisperModel; \
    WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
