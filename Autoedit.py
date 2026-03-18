import os
import whisper
import numpy as np
import requests
import zipfile
import io
import json
import argparse
from datetime import timedelta
from dotenv import load_dotenv, set_key
from google import genai
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from moviepy.audio.fx import AudioFadeIn, AudioFadeOut

# --- CONFIGURATION & ENV ---
ENV_PATH = ".env"
load_dotenv(ENV_PATH)
FONT_DIR = "fonts"
SHORTS_DIR = "shorts"
# The user manually provides the font: "orange juice 2.0.ttf"
FONT_FILENAME = "orange juice 2.0.ttf"


def setup_assets():
    """Ensure shorts directory and font are ready."""
    if not os.path.exists(SHORTS_DIR):
        os.makedirs(SHORTS_DIR)
    
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)
        
    font_path = os.path.join(FONT_DIR, FONT_FILENAME)
    
    print("\n--- FONT LICENSING NOTICE ---")
    print("The 'Orange Juice' typeface is FREE for non-commercial work only.")
    print("For commercial use, please pay $5 at www.brittneymurphydesign.com.")
    print("-----------------------------\n")

    if not os.path.exists(font_path):
        print(f"-> Error: Font not found at {font_path}")
        print(f"Please place '{FONT_FILENAME}' in the '{FONT_DIR}' directory.")
        raise FileNotFoundError(f"Missing font: {font_path}")
        
    return font_path


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n--- GEMINI API SETUP ---")
        api_key = input("Please enter your Gemini API Key (it will be saved to .env for future use): ").strip()
        set_key(ENV_PATH, "GEMINI_API_KEY", api_key)
    return genai.Client(api_key=api_key)


def detect_hooks_and_speakers(video_path, transcript):
    """Analyze video with Gemini to find viral hooks and speaker layouts."""
    print("\n[AI] Analyzing video for viral hooks and speaker layout...")
    client = get_gemini_client()

    # Upload video for analysis
    file = client.files.upload(file=video_path)
    print(f"-> Video uploaded. ID: {file.name}")

    # Prompt for Viral Hook Detection
    prompt = f"""
    Analyze this video and the following transcript. 
    Find 3 viral hooks (each between 15-60 seconds long) that are punchy and stand alone well.
    For EACH hook, identify the speaker(s) and their layout.
    
    If there are multiple speakers, tell me where they are positioned in the frame (e.g. "Speaker 1: Left, Speaker 2: Right").
    If it's a single speaker, just say "Center".
    
    Return ONLY a JSON object with this exact structure:
    {{
        "hooks": [
            {{
                "start_time": float,
                "end_time": float,
                "title": "Short title",
                "active_speaker_map": [
                    {{"time": float, "speaker": "Speaker Name", "position": "left|right|center"}}
                ]
            }}
        ]
    }}
    
    TRANSCRIPT:
    {transcript}
    """

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[file, prompt],
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    return json.loads(response.text)


def apply_vertical_crop(clip, position="center"):
    """Crop a horizontal clip to vertical (9:16) focusing on a position."""
    w, h = clip.size
    target_w = h * 9 / 16
    
    if position == "left":
        x1 = 0
    elif position == "right":
        x1 = w - target_w
    else: # center
        x1 = (w - target_w) / 2
        
    return clip.cropped(x1=x1, y1=0, width=target_w, height=h).resized(width=1080)
# --- UTILITIES ---
def format_srt_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def save_srt(segments, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start = format_srt_time(seg['start'])
            end = format_srt_time(seg['end'])
            f.write(f"{i + 1}\n{start} --> {end}\n{seg['text'].strip()}\n\n")


def generate_viral_shorts(video_path, ai_hooks, whisper_result, font_path):
    """Generate multiple viral shorts from the detected hooks."""
    print(f"\n[4/5] Generating {len(ai_hooks['hooks'])} Viral Shorts...")
    video = VideoFileClip(video_path)
    colors = ["#5BCEFA", "#F5A9B8", "#FFFFFF"] # Trans Colors: Blue, Pink, White
    
    for i, hook in enumerate(ai_hooks['hooks']):
        print(f"-> Processing Short {i+1}: {hook['title']}")
        start_t = hook['start_time']
        end_t = hook['end_time']
        
        # Subclip for this hook
        hook_clip = video.subclipped(start_t, end_t)
        
        # 1. Dynamic Cropping (Speaker Focus)
        # Split clip into segments based on speaker changes
        segments = []
        speaker_map = hook.get('active_speaker_map', [])
        
        if not speaker_map:
            # Default to center if no map provided
            final_clip = apply_vertical_crop(hook_clip, "center")
        else:
            # Build segments based on speaker positions
            for j in range(len(speaker_map)):
                seg_start = speaker_map[j]['time'] - start_t
                seg_end = (speaker_map[j+1]['time'] - start_t) if j+1 < len(speaker_map) else (end_t - start_t)
                
                if seg_end > seg_start:
                    seg_clip = hook_clip.subclipped(max(0, seg_start), min(hook_clip.duration, seg_end))
                    seg_clip = apply_vertical_crop(seg_clip, speaker_map[j]['position'])
                    segments.append(seg_clip)
            
            final_clip = concatenate_videoclips(segments) if segments else apply_vertical_crop(hook_clip, "center")

        # 2. Add Captions (Orange Juice font + Trans Colors)
        all_words = []
        for segment in whisper_result['segments']:
            if 'words' in segment:
                all_words.extend(segment['words'])
        
        # Filter words within this hook's timeframe
        hook_words = [w for w in all_words if w['start'] >= start_t and w['end'] <= end_t]
        
        caption_clips = []
        color_idx = 0
        for w in hook_words:
            # Create a TextClip for each word (or small phrase)
            # MoviePy 2.x TextClip syntax
            txt = TextClip(
                text=w['word'].strip().upper(),
                font=font_path,
                font_size=90,
                color=colors[color_idx % 3],
                stroke_color="black",
                stroke_width=2,
                method='label',
                duration=w['end'] - w['start']
            ).with_start(w['start'] - start_t).with_position(("center", 1400)) # Bottom-ish
            
            caption_clips.append(txt)
            color_idx += 1
            
        # Composite video and captions
        final_short = CompositeVideoClip([final_clip] + caption_clips)
        
        # Write output
        output_path = os.path.join(SHORTS_DIR, f"short_{i+1}_{hook['title'].replace(' ', '_')}.mp4")
        final_short.write_videofile(output_path, fps=video.fps, threads=4, preset="ultrafast")
        print(f"   -> Saved to: {output_path}")

    video.close()


# --- CORE PIPELINE ---
def run_full_pipeline(input_path, output_video_name, min_duration=2.0, do_edit=True, do_shorts=True, do_metadata=True):
    # 0. Setup Assets
    font_path = setup_assets()
    
    # 1. Transcription (Whisper with Progress Bar)
    # This is needed for editing, shorts, and metadata
    print("\n[1/5] Starting Transcription...")
    model = whisper.load_model("base")

    result = model.transcribe(
        input_path,
        fp16=False,
        verbose=True,
        language='en'
    ) # Remove word_timestamps if it causes errors, or check version.
    # If the user has a version that doesn't support it in the transcribe() signature
    # but supports it internally, we might need a different approach.
    # However, for most recent versions, it IS a valid argument for transcribe.
    # Let's try to remove it from the direct call if it's failing at the DecodingOptions level.
    # Wait, the error is inside transcribe.py calling decode_with_fallback.
    # It seems the installed whisper version has an issue with this parameter.


    # 2. Neuro-Inclusive Video Editing (Main Master)
    if do_edit:
        print(f"\n[2/5] Processing Main Master (Min Clip Length: {min_duration}s)...")
        video = VideoFileClip(input_path)

        # Filter fillers and merge for ADHD/Autism friendly pacing
        blacklist = ["um", "uh", "ah", "erm", "uhm"]
        raw_segs = [s for s in result['segments'] if s['text'].lower().strip(".,!? ") not in blacklist]

        merged = []
        if raw_segs:
            s, e = raw_segs[0]['start'], raw_segs[0]['end']
            for i in range(1, len(raw_segs)):
                nxt = raw_segs[i]
                if (nxt['start'] - e < 0.4) or (e - s < min_duration):
                    e = nxt['end']
                else:
                    merged.append((s, e))
                    s, e = nxt['start'], nxt['end']
            merged.append((s, e))

        final_clips = []
        zoom_state = False

        for start_t, end_t in merged:
            clip = video.subclipped(max(0, start_t - 0.05), min(video.duration, end_t + 0.05))

            # Audio Fades (MoviePy 2.0 Syntax)
            if clip.audio is not None:
                clip = clip.with_effects([AudioFadeIn(0.1), AudioFadeOut(0.1)])

            # Zoom Logic
            if zoom_state and (end_t - start_t > 1.5):
                w, h = clip.size
                clip = clip.cropped(x1=w * 0.07, y1=h * 0.07, width=w * 0.86, height=h * 0.86).resized(width=w)
                zoom_state = False
            else:
                zoom_state = True
            final_clips.append(clip)

        if final_clips:
            final_video = concatenate_videoclips(final_clips, method="compose", padding=-0.05)
            # Audio Mastering (Sensory-friendly normalization)
            if final_video.audio is not None:
                max_v = final_video.audio.max_volume()
                if max_v > 0:
                    final_video = final_video.with_audio(final_video.audio.with_volume_scaled(0.65 / max_v))
            final_video.write_videofile(output_video_name, threads=4, fps=video.fps, preset="ultrafast")

            # 3. Redo Subtitles at the End
            print("\n[3/5] Generating Synced Subtitles for Final Master...")
            final_result = model.transcribe(
                output_video_name,
                fp16=False,
                verbose=False,
                language='en'
            )
            final_srt_path = output_video_name.replace(".mp4", ".srt")
            save_srt(final_result['segments'], final_srt_path)
            print(f"-> Synced subtitles saved to: {final_srt_path}")
        else:
            print("-> Warning: No clips found for main master.")
    else:
        print("\n[2/5] Skipping Video Editing step.")
        print("[3/5] Skipping Synced Subtitles step.")

    # 4. Viral Shorts Generation (Optional)
    if do_shorts:
        try:
            ai_hooks = detect_hooks_and_speakers(input_path, result['text'])
            print(f"\n[4/5] Generating {len(ai_hooks['hooks'])} Viral Shorts...")
            generate_viral_shorts(input_path, ai_hooks, result, font_path)
        except Exception as e:
            print(f"-> Error generating shorts: {e}")
    else:
        print("\n[4/5] Skipping Viral Shorts Generation.")

    # 5. Gemini 3.0 Metadata Generation
    if do_metadata:
        print("\n[5/5] Generating Metadata with Gemini 3 Flash...")
        client = get_gemini_client()

        prompt = f"""
        Take the following transcript, and generate a youtube title and description for it as well as a single word or short phrase (up to 20 characters) that sums up the video.
        Additionally provide me with text for a BlueSky post to promote the video.

        PERSONA:
        - Direct, Unfiltered, Punchy sentences. No 'fluff'.
        - Principled Advocate (Bodily autonomy/Trans rights). Sharp on hypocrisy.
        - Dry British Humor. Understated irony. Not 'bubbly'.
        - 49-year-old software developer's grounded, blunt perspective.

        RULES:
        - Title: <60 chars, include 2026, no clickbait.
        - Description: Keywords at start (max 5). 1000-3000 chars. 
        - Popular Language: Explain how this affects everyone, not just trans people.
        - DO NOT HALLUCINATE: Only use facts from the transcript.

        TRANSCRIPT:
        {result['text']}
        """

        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt
        )

        meta_path = "youtube_metadata.txt"
        with open(meta_path, "w") as f:
            f.write(response.text)

        print(f"-> Metadata saved to: {meta_path}")
    else:
        print("\n[5/5] Skipping Metadata Generation.")

    print("\nDONE. Everything is ready for upload.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
ADHD/Autism Friendly Video Auto-Editor
--------------------------------------
This script processes raw video to:
1. Trim silences and filler words (um, uh, etc.) for punchy pacing.
2. Apply sensory-friendly audio normalization.
3. Generate synced subtitles for the edited master.
4. Detect viral hooks and generate vertical (9:16) shorts with captions.
5. Generate YouTube titles, descriptions, and BlueSky promotional text via Gemini AI.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Autoedit.py input.mp4                      # Full pipeline
  python Autoedit.py input.mp4 --no-shorts          # Skip viral shorts
  python Autoedit.py input.mp4 --min-duration 1.5   # Set minimum clip length
  python Autoedit.py input.mp4 --no-edit --no-metadata # Only generate shorts
"""
    )
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("--output", help="Output filename for edited video (defaults to input_finalised.mp4)")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum clip duration for editing (default: 2.0s)")
    parser.add_argument("--no-edit", action="store_true", help="Skip the video editing and synced subtitle generation")
    parser.add_argument("--no-shorts", action="store_true", help="Skip viral shorts generation")
    parser.add_argument("--no-metadata", action="store_true", help="Skip metadata generation")

    args = parser.parse_args()

    # Default output logic: {original_path}_finalised.mp4
    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_finalised.mp4"

    run_full_pipeline(
        args.input, 
        output_path, 
        min_duration=args.min_duration,
        do_edit=not args.no_edit,
        do_shorts=not args.no_shorts,
        do_metadata=not args.no_metadata
    )
