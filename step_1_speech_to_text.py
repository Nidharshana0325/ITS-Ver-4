import whisper
import json
import subprocess
from pathlib import Path
import sys

# =========================
# PATH SETUP
# =========================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

AUDIO_PATH = DATA_DIR / "input_audio.wav"
OUTPUT_PATH = DATA_DIR / "transcript.json"


# =========================
# DOWNLOAD AUDIO (FIXED)
# =========================

def download_youtube_audio(url: str):
    """
    Downloads BEST audio from YouTube and converts to WAV.
    This avoids DASH video-only streams and Whisper FFmpeg pipe errors.
    """
    print("⬇️ Downloading YouTube audio...")

    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(AUDIO_PATH),
        url
    ]

    subprocess.run(command, check=True)
    print("✅ Audio downloaded successfully")


# =========================
# TRANSCRIBE AUDIO
# =========================

def transcribe_audio(audio_path: Path):
    print("🧠 Loading Whisper model...")
    model = whisper.load_model("small")  # accurate + stable on CPU

    print("📝 Transcribing audio (this may take a few minutes)...")

    result = model.transcribe(
        str(audio_path),
        verbose=False
    )

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print("✅ Transcription complete!")
    print(f"📄 Saved transcript to: {OUTPUT_PATH}")


# =========================
# MAIN ENTRY
# =========================

def main():
    print(
        "\nChoose input type:\n"
        "1. YouTube link\n"
        "2. Local audio/video file\n"
    )

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        url = input("Enter YouTube URL: ").strip()
        download_youtube_audio(url)

    elif choice == "2":
        local_path = Path(input("Enter local file path: ").strip())

        if not local_path.exists():
            print("❌ File not found.")
            sys.exit(1)

        # Copy local file to expected location
        AUDIO_PATH.write_bytes(local_path.read_bytes())
        print("✅ Local file copied")

    else:
        print("❌ Invalid choice.")
        sys.exit(1)

    transcribe_audio(AUDIO_PATH)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    main()
