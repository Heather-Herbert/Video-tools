# Video-tools

Python tooling for The Polycule's YouTube production. Raw camera file in; a
Kdenlive project, a thumbnail and a metadata text file out.

Everything up to rendering is automatic. Rendering is not — you check the edit
first, and the pipeline refuses to render while anything is flagged.

```
raw MVI_1234.MP4
        │
        ├─ ingest      probe, 720p proxy, 16 kHz audio
        ├─ transcribe  faster-whisper (+ speaker labels if pyannote is set up)
        ├─ analyse     LLM → cuts, chapters, cards, punch-ins
        ├─ graphics    render the cards as transparent PNGs
        ├─ metadata    title, description, tags, Bluesky draft
        ├─ assemble    write the .kdenlive project
        └─ review      list what a human must sign off
                │
      you pick a thumbnail frame, clear the review items
                │
        └─ render      (optional — you may prefer to finish by hand)

outputs:  2026-09-06.kdenlive
          2026-09-06-thumb.png
          2026-09-06-metadata.txt
```

Kdenlive is the editor of record because its project files are MLT XML a script
can write directly. The pipeline is not committed to it: every stage before
assembly produces a tool-neutral edit decision list, and only
`writers/kdenlive.py` knows what an editor is. A Resolve/FCPXML writer can be
added without touching anything upstream.

---

## Setup, from a fresh Kubuntu install

Tested on Kubuntu 24.04 and later. Copy-paste each block.

### 1. System packages

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv git \
    ffmpeg \
    kdenlive \
    melt \
    fontconfig
```

- **ffmpeg** — proxies, audio extraction, frame grabs. Required.
- **kdenlive** — to open and finish the project. Required in practice.
- **melt** — only for headless `render`. Skip it if you always render in the
  Kdenlive GUI.

### 2. The brand typeface

Graphics fall back to DejaVu without it and render off-brand. The stage warns
when this happens, but the warning is easy to miss.

```bash
mkdir -p ~/.local/share/fonts
cd ~/.local/share/fonts
curl -L -o fredoka.zip "https://fonts.google.com/download?family=Fredoka"
unzip -o fredoka.zip 'static/*' -d fredoka && rm fredoka.zip
fc-cache -f
fc-match "Fredoka:style=SemiBold"     # should name a Fredoka file, not DejaVu
```

### 3. The repo and its Python dependencies

```bash
git clone https://github.com/Heather-Herbert/Video-tools.git
cd Video-tools

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

That pulls `mediapipe` and `rembg` (plus `onnxruntime`), which together are
about 500 MB. They are only needed for the thumbnail stage — if you always pick
frames by hand, comment them out of `requirements.txt`.

`rembg` downloads its model on first use, to `~/.u2net/`. The first cutout is
therefore slow and needs a network connection; every later one is local.

### 4. Configuration

```bash
cp .env.example .env
$EDITOR .env
```

`.env` is gitignored. Never commit real keys.

You need **at least one** working LLM backend. The chain tries them in order
and stops at the first that answers:

| Backend | Cost | How to enable |
|---|---|---|
| Claude CLI | included in a Claude subscription | install the `claude` CLI, run `claude` once to log in |
| `agy` CLI | free tier | install Antigravity, make `agy` available on PATH |
| DeepSeek | paid, cheap | set `DEEPSEEK_API_KEY` |
| OpenRouter | paid | set `OPENROUTER_API_KEY` |

If you have `claude` or `agy` on PATH, you can leave both keys blank.

Set `POLYCULE_EPISODES` to where episodes should live. Budget roughly three
times the size of the raw file per episode, for proxies and extracted audio.

### 5. Optional: speaker labels

Without this the pipeline still runs, but it cannot tell Heather from Sophie,
so name cards and per-speaker punch-ins are skipped and the segments are
flagged for you to label.

```bash
pip install "pyannote.audio"
```

Then accept the model licence at
<https://hf.co/pyannote/speaker-diarization-3.1>, create a read token at
<https://hf.co/settings/tokens>, and put it in `.env` as `HF_TOKEN`.

`pyannote` pulls in PyTorch — about 2 GB.

### 6. Check it works

```bash
python -m pytest            # 64 tests, no media, models or API keys needed
python -m video_pipeline.run --help
```

---

## Use

```bash
source .venv/bin/activate

# everything up to the review gate
python -m video_pipeline.run all --slug 2026-09-06 --raw ~/footage/MVI_1234.MP4

# pick a thumbnail: writes a contact sheet of vetted frames
python -m video_pipeline.run thumbnail --slug 2026-09-06
xdg-open ~/episodes/2026-09-06/work/thumbnail/contact-sheet.png
python -m video_pipeline.run thumbnail --slug 2026-09-06 --pick 3

# read what needs checking, then clear it
python -m video_pipeline.run review --slug 2026-09-06
python -m video_pipeline.run review --slug 2026-09-06 --resolve 0 1 2

# open the project and finish by hand (the usual route), or render headless
kdenlive ~/episodes/2026-09-06/2026-09-06.kdenlive
python -m video_pipeline.run render --slug 2026-09-06
```

Every stage is independent and re-runnable. They all read and write
`<slug>/<slug>.edl.json`, so you can fix the analysis and re-run from `analyse`
without paying for the transcript again.

| Stage | Does | Slow? |
|---|---|---|
| `ingest` | probe, 720p proxy, 16 kHz audio | yes, once |
| `transcribe` | faster-whisper + optional diarization | yes (`--remote`) |
| `analyse` | LLM → cuts, chapters, cards, punch-ins | ~1 call |
| `graphics` | render cards as transparent PNGs | fast, cached |
| `thumbnail` | score frames, then cut out and halo the one you pick | medium (`--remote`) |
| `metadata` | title, description, tags, Bluesky draft | ~1 call |
| `assemble` | write the `.kdenlive` project | fast |
| `review` | list / resolve the human checkpoint | — |
| `render` | melt render + two-pass loudnorm | yes |

`all` deliberately stops before `thumbnail`, because that stage needs you to
choose a frame.

---

## The thumbnail stage

Two steps, with you in the middle.

**Choosing a frame.** The script cannot tell which moment sets the tone of a
video — nothing in the pixels carries that. What it can do is narrow the search
to moments the analysis already judged important (pull-quotes, reactions,
chapter openings) and then reject frames that are technically unusable: blinks,
mid-word mouth shapes, motion blur, faces turned away. It writes a contact
sheet of the survivors and stops.

The score filters out disqualifying faults; it does **not** rank thumbnails by
quality. Left to rank freely it prefers neutral expressions, and neutral makes
a dull thumbnail. Pick from the shortlist with your own eye.

You can also skip the search: `--at 252` scores frames around a timestamp you
already have in mind, and `--frame path.png` uses an image directly.

**The halo.** The Legal Eagle sticker outline is a hard-edged stroke, not a
glow, and two things produce it:

1. **A binary alpha.** Background removal leaves a soft, partly transparent
   edge. Growing a soft matte spreads it like a blur, and every corner and
   hair-tip rounds off into a blob — this is why a feathered "grow selection"
   in GIMP gives a rounded halo rather than a die-cut one. The stage thresholds
   the alpha to 0/255 first, and erodes a pixel to strip the fringe of leftover
   background colour.
2. **Constant-width dilation.** A morphological grow with a square structuring
   element, which holds sharp corners. Never a Gaussian glow.

The stack is composed at 2× and scaled down, so the edge is antialiased without
the shape ever being softened.

The default strokes read outward from the subject as **white, pink, blue** —
the trans flag's own order — at 10, 20 and 30px. The halo hugs the speakers
rather than bordering the frame: `SUBJECT_HEIGHT` (0.80) keeps the subject to
four-fifths of the frame height so the whole outline stays visible instead of
running off the edges. All of it, plus `SHADOW`, is at the top of
`stages/thumbnail.py`.

The text block colour is `THUMB_ACCENT` and is deliberately kept out of
`brand.py`: on-screen cards stay trans-flag palette, and the thumbnail block
does not.

Drop your existing template artwork at `assets/templates/thumbnail.png` (1280×720)
and it is used as the background; without one the stage paints a flat brand
background so it still works on a fresh machine.

---

## Running stages on RunPod

Two stages are worth renting a GPU for on a machine short of graphics power:
transcription, and the thumbnail's background removal and face scoring.
Everything else — analysis, card rendering, XML assembly — is cheap and stays
local.

Remote is opt-in per run and never required:

```bash
python -m video_pipeline.run transcribe --slug 2026-09-06 --remote
python -m video_pipeline.run thumbnail  --slug 2026-09-06 --remote
```

Without `--remote` nothing changes, and with it but no configuration the stage
stops and tells you rather than failing deep inside a library.

A cold worker takes 30-60s to boot, so a short clip is often faster locally.
The win is on a full episode: `large-v3` on a rented GPU against `base` on a
CPU is both faster and more accurate.

### One-time setup

**1. Object storage.** RunPod's job payloads are far too small for audio, so
media goes through an S3-compatible bucket and the job carries only keys.

**Cloudflare R2 is the one to use.** This workload is egress-shaped — the
worker downloads your audio out of the bucket on every job — and R2 charges
nothing for egress, where B2 and S3 both bill it. The free tier (10 GB) is far
more than this needs, since the bucket only ever holds intermediates that are
deleted on success.

1. Cloudflare dashboard → R2 → **Create bucket**.
2. → **Manage R2 API Tokens** → **Create token**, permission *Object Read &
   Write*, scoped to that bucket. The Access Key ID and Secret are shown once.
3. The endpoint URL is on the bucket's settings page, in the form
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`.
4. Optional: bucket → Settings → Object lifecycle rules → delete after 1 day,
   as a backstop to the pipeline's own cleanup.

```bash
S3_BUCKET=video-tools-scratch
S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=<from the token>
S3_SECRET_ACCESS_KEY=<from the token>
S3_REGION=auto           # R2 does not use AWS-style regions
```

Backblaze B2, AWS S3 and a RunPod network volume with the S3 gateway all work
too — set the same five variables.

**2. Build and push the worker image.**

```bash
docker build -t <dockerhub-user>/video-tools-worker:1 runpod_worker/
docker push  <dockerhub-user>/video-tools-worker:1
```

The image is around 6 GB: Whisper large-v3 and the rembg model are baked in
rather than downloaded on every cold start.

**3. Create a serverless endpoint** on that image at
<https://runpod.io/console/serverless>, with a 16 GB GPU, 15 GB container disk,
max workers 1, idle timeout 5s and FlashBoot on. Give it these environment
variables: `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`,
`S3_SECRET_ACCESS_KEY`, `S3_REGION`, and `HF_TOKEN` if you want speaker labels.

**4. Fill in `.env` on the desktop:**

```
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...        # from the endpoint page
S3_BUCKET=...
S3_ENDPOINT_URL=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
```

```bash
pip install boto3             # only needed for --remote
```

### How it behaves

- Jobs are submitted **asynchronously** and polled. A synchronous request times
  out at the HTTP layer long before a 45-minute transcript finishes, and the
  job then runs on invisibly — billed and unreachable.
- If you stop waiting, the job is **cancelled**, so a hung job does not keep
  billing a GPU with nobody left to collect the result.
- Uploaded audio is deleted from the bucket afterwards, including when the job
  fails.
- boto3 1.36 and later send CRC32 integrity checksums on every upload by
  default, which **R2 rejects** with an opaque `x-amz-content-sha256 is
  invalid`. Both clients set `request_checksum_calculation="when_required"` to
  restore the old behaviour. If you ever rewrite the S3 client, keep that.
- The worker returns **raw measurements** for frame scoring, not verdicts. The
  thresholds that decide what counts as a blink live on the desktop in
  `stages/thumbnail.py`, so tuning them never means rebuilding a 6 GB image.

Rendering is not offloaded. It is MLT/melt work rather than GPU work, and you
usually want to finish the edit by hand in Kdenlive anyway.

## The metadata file

`<slug>-metadata.txt` holds the title, thumbnail phrase, description, chapter
timestamps, tags and a Bluesky post — laid out for a human with the YouTube
upload form open.

**Nothing is posted and nothing is uploaded.** The Bluesky draft is written
with a literal `<<PASTE VIDEO LINK>>` placeholder, and any URL the model
invents is stripped, so an unfinished post is obviously unfinished rather than
quietly linkless.

## Voice

`voice.py` holds the channel promise, the writing persona and the guardrails,
and every prompt that generates audience-facing words reads from it. It lives in
the repo rather than in anyone's head, because this runs on a desktop with no
access to either.

Two sources feed it and they pull in different directions: the channel promise
("positive, funny, refusing to give up — no rage bait") and the persona carried
over from `Autoedit.py` ("direct, unfiltered, sharp on hypocrisy"). They are
reconciled explicitly rather than left to fight inside a prompt: **sharp about
power, warm toward people.** Anger at a policy is on-voice; despair aimed at the
audience is not. Change one and read the other.

---

## The two clocks

The single biggest source of bugs here. The LLM reads a transcript, so
everything it says is in **raw time** — seconds into the camera file. Cards and
punch-ins have to land in **timeline time** — seconds into the finished edit,
after cuts are removed. `Timeline.conform()` does that translation exactly
once. Nothing downstream re-maps anything, and the writer refuses an unconformed
timeline.

The thumbnail stage runs after conform, so it maps back the other way
(`timeline_to_raw`) to address the original camera file.

## The review gate

`analyse` flags every citation and statistic; `metadata` flags every generated
title, Bluesky post and factual claim. `render` refuses while any blocking item
is open.

Source cards and channel copy make claims in the channel's name. They do not
auto-publish. `--force` exists, but use it only when you know why.

## MLT behaviours worth knowing

Each of these was found by rendering and looking at the result. They are locked
in by `tests/test_writer.py`.

- **`qtblend` transitions drop the upper track's alpha** in a headless melt
  render, so a transparent graphics card punches through to black and hides the
  camera. Video tracks use `composite` instead.
- **`qtblend` filters are inert** in this MLT build — a "Transform" effect
  written into the XML does nothing.
- **`composite` geometry keyframes are normalised to the transition length**, so
  a keyframed punch-in smears across the whole episode. Punch-ins are therefore
  separate tracks, one per framing, each with static geometry.
- **melt renders whatever `<mlt producer=...>` names**, which in a Kdenlive file
  is the project bin, not the timeline; and `black_track` is 2³¹ frames long, so
  a render with no explicit frame range never ends. `render._render_target`
  fixes both on a temp copy.

## Layout

```
video_pipeline/
  edl.py               tool-neutral edit decision list + the time mapping
  brand.py             palette and typeface, from the vault style guide
  llm.py               backend chain and the only reader of .env
  voice.py             channel promise, persona and guardrails, in one place
  remote.py            RunPod offload for the GPU-heavy stages
  stages/
    ingest.py          probe, proxy, audio extraction, silence detection
    transcribe.py      faster-whisper + optional pyannote
    analyse.py         the LLM pass; everything structural stays in code
    graphics.py        ten card renderers, content-hashed and cached
    thumbnail.py       frame scoring, cutout, sticker halo, template
    metadata.py        title, description, tags, Bluesky draft
    render.py          loudnorm + headless melt
  writers/
    kdenlive.py        MLT XML
  run.py               stage CLI
runpod_worker/
  handler.py           the serverless worker; imports nothing from the pipeline
  Dockerfile           CUDA image with Whisper and rembg baked in
legacy/
  Autoedit.py          the previous all-in-one script
  RemoveBackground.py  standalone rembg wrapper
```

`legacy/` is kept working but is no longer developed. `Autoedit.py` still has
the viral-shorts generator, which the pipeline does not.

## Not built yet

- Intro/outro stings, SFX library, subscribe animation — the timeline supports
  `sfx` and `stinger` sources, but there are no assets yet.
- Viral shorts as a pipeline stage (still only in `legacy/Autoedit.py`).
- n8n trigger on a watched folder.
- Demucs source separation (untested for a single mixed track).
