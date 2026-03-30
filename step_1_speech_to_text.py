"""
STEP 1 — Speech to Text
────────────────────────
Transcribes YouTube URLs or local audio/video files using OpenAI Whisper.
Produces word-level timestamps for precise sync with board/mindmap.

Output: data/transcript.json
  Each segment: { start, end, text, words: [{word, start, end}] }
"""

import whisper
import json
import subprocess
from pathlib import Path
import sys

DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
AUDIO_PATH = DATA_DIR / "input_audio.wav"
OUTPUT     = DATA_DIR / "transcript.json"


def download_youtube(url: str):
    print("⬇️  Downloading YouTube audio...")
    subprocess.run([
        "yt-dlp", "-f", "bestaudio",
        "--extract-audio", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(AUDIO_PATH), url
    ], check=True)
    print("✅ Audio downloaded")


def transcribe(audio_path: Path):
    print("🧠 Loading Whisper model (small)...")
    model = whisper.load_model("small")
    print("📝 Transcribing with word timestamps...")

    result = model.transcribe(
        str(audio_path),
        verbose=False,
        word_timestamps=True   # enables per-word timing
    )

    segments = []
    for seg in result["segments"]:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word":  w["word"].strip(),
                "start": round(w["start"], 3),
                "end":   round(w["end"],   3),
            })
        segments.append({
            "start": round(seg["start"], 2),
            "end":   round(seg["end"],   2),
            "text":  seg["text"].strip(),
            "words": words,
        })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    total_words = sum(len(s["words"]) for s in segments)
    print(f"✅ Transcript saved → {OUTPUT}")
    print(f"   {len(segments)} segments | {total_words} words with timestamps")


def main():
    print("\n=== STEP 1: Speech to Text ===\n")
    print("Choose input:\n  1. YouTube link\n  2. Local audio/video file\n")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        url = input("YouTube URL: ").strip()
        download_youtube(url)
    elif choice == "2":
        path = Path(input("Local file path: ").strip())
        if not path.exists():
            print("❌ File not found."); sys.exit(1)
        import shutil
        shutil.copy(path, AUDIO_PATH)
        print("✅ File copied")
    else:
        print("❌ Invalid choice."); sys.exit(1)

    transcribe(AUDIO_PATH)


if __name__ == "__main__":
    main()