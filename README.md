# Video AI Toolkit

A collection of AI-powered tools for neuro-inclusive video editing, automated subtitle generation, viral short creation, and background removal.

## 🛠 Features

### 1. ADHD/Autism Friendly Auto-Editor (`Autoedit.py`)
A comprehensive pipeline designed for high-pacing and sensory-friendly video production.
- **Silence Removal:** Automatically cuts silences and filler words (um, uh, etc.).
- **Dynamic Zoom:** Adds subtle visual variety to maintain engagement.
- **Sensory-Friendly Audio:** Normalizes volume to comfortable levels.
- **Synced Subtitles:** Generates perfectly timed `.srt` files for the edited master.
- **Viral Shorts:** Uses Gemini AI to find "hooks" and generates 9:16 vertical clips with stylized captions.
- **AI Metadata:** Generates YouTube titles, descriptions, and BlueSky posts using Gemini 2.0/3.0.

### 2. Background Remover (`RemoveBackground.py`)
- Quickly strips backgrounds from images (PNG/JPG) using the `rembg` library.
- Ideal for creating thumbnails or speaker overlays.

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **FFmpeg:** Required for video/audio processing.
  - *Linux:* `sudo apt install ffmpeg`
  - *macOS:* `brew install ffmpeg`
  - *Windows:* Install via [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/)

### Installation
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # OR
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## 📖 Usage

### Autoedit
Run the full pipeline or select specific stages:
```bash
# Full process
python Autoedit.py input.mp4

# Only generate metadata and shorts (skip editing)
python Autoedit.py input.mp4 --no-edit

# Only perform the video edit
python Autoedit.py input.mp4 --no-shorts --no-metadata
```

### Background Removal
```bash
python RemoveBackground.py
```
*(Update the `target_image` path in the script or modify it to accept arguments).*

---

## 📝 License
MIT
