# runpod_worker

The container that runs on RunPod. Nothing here imports `video_pipeline`: the
worker's only contract is the JSON it returns, so the image and the pipeline
version independently.

| Stage | Input | Output |
|---|---|---|
| `transcribe` | S3 key for a 16 kHz wav | `{segments, language, text, speakers?}` |
| `cutout` | base64 PNG | `{image_b64}` |
| `score` | base64 PNGs | `{scores: [{eye, mouth, yaw, sharpness, face_h}]}` |

`score` returns raw measurements, not a verdict. The thresholds that decide
what counts as a blink live on the desktop in `stages/thumbnail.py`, so tuning
them does not mean rebuilding and pushing a 6 GB image.

## Build

```bash
docker build -t <you>/video-tools-worker:1 runpod_worker/
docker push  <you>/video-tools-worker:1
```

The image is large (~6 GB) because Whisper large-v3 and the rembg model are
baked in rather than downloaded per cold start.

## Endpoint settings

Create a serverless endpoint on that image with:

- GPU: 16 GB (RTX 4000 Ada or better). large-v3 needs about 10 GB.
- Container disk: 15 GB.
- Max workers: 1 is plenty for one person's episodes.
- Idle timeout: 5s. You pay for idle time, and jobs here are minutes apart.
- FlashBoot: on. It keeps the model in memory between nearby jobs.

Environment variables on the endpoint:

```
S3_BUCKET, S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_REGION
HF_TOKEN     # only if you want speaker labels
```
