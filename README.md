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
python -m pytest            # 53 tests, no media or API keys needed
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
| `transcribe` | faster-whisper + optional diarization | yes |
| `analyse` | LLM → cuts, chapters, cards, punch-ins | ~1 call |
| `graphics` | render cards as transparent PNGs | fast, cached |
| `thumbnail` | score frames, then cut out and halo the one you pick | medium |
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
the shape ever being softened. Widths and colours are `DEFAULT_STROKES` and
`SHADOW` at the top of `stages/thumbnail.py`; the text block colour is
`THUMB_ACCENT`, kept out of `brand.py` because on-screen cards stay trans-flag
palette and the thumbnail block does not.

Drop your existing template artwork at `assets/templates/thumbnail.png` (1280×720)
and it is used as the background; without one the stage paints a flat brand
background so it still works on a fresh machine.

---

## The metadata file

`<slug>-metadata.txt` holds the title, thumbnail phrase, description, chapter
timestamps, tags and a Bluesky post — laid out for a human with the YouTube
upload form open.

**Nothing is posted and nothing is uploaded.** The Bluesky draft is written
with a literal `<<PASTE VIDEO LINK>>` placeholder, and any URL the model
invents is stripped, so an unfinished post is obviously unfinished rather than
quietly linkless.

The persona and rules in the prompt came from the older `legacy/Autoedit.py`
and were tuned by hand. Edit them deliberately — they are the reason the output
sounds like Heather and not like a model.

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
legacy/
  Autoedit.py          the previous all-in-one script
  RemoveBackground.py  standalone rembg wrapper
```

`legacy/` is kept working but is no longer developed. `Autoedit.py` still has
the viral-shorts generator, which the pipeline does not.

## Not built yet

- RunPod serverless endpoints for the GPU stages (`transcribe`, and the
  thumbnail's rembg/mediapipe pass). Every stage round-trips through
  `<slug>.edl.json`, so a remote stage is "ship the work dir up, run one stage,
  ship the delta back" rather than a re-architecture.
- Intro/outro stings, SFX library, subscribe animation — the timeline supports
  `sfx` and `stinger` sources, but there are no assets yet.
- Viral shorts as a pipeline stage (still only in `legacy/Autoedit.py`).
- n8n trigger on a watched folder.
- Demucs source separation (untested for a single mixed track).
