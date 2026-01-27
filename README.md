📦 Project Requirements

This document lists all dependencies and setup steps required to run the full 7-step AI tutoring pipeline.

🐍 Python Version
Python >= 3.9 and <= 3.11


⚠️ Python 3.12 is not recommended due to compatibility issues with some libraries.

📚 Python Packages

Install all required Python packages using pip:

pip install \
torch \
torchvision \
torchaudio \
numpy \
scikit-learn \
sentence-transformers \
openai-whisper \
llama-cpp-python \
yt-dlp \
tqdm

🛠 System / External Dependencies
FFmpeg (Mandatory)

Required for audio processing used by Whisper.

Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

macOS (Homebrew)
brew install ffmpeg

Windows

Download from: https://ffmpeg.org/download.html

Add ffmpeg to your system PATH

Verify installation:

ffmpeg -version

🧠 Model Files (Manual Download Required)

Download the following model manually and place it exactly as shown:

models/Phi-3.5-mini-instruct-Q4_K_L.gguf


Source: HuggingFace
(Search for: Phi-3.5-mini-instruct GGUF)

Recommended quantization: Q4_K_L (best for CPU usage)

📥 Auto-Downloaded Models

The following models are automatically downloaded on first run (no action needed):

Whisper model: small

SentenceTransformer model: all-MiniLM-L6-v2

📁 Required Folder Structure

Create the required directories before running the pipeline:

mkdir -p data models


Expected structure:

project_root/
├── data/
├── models/
│   └── Phi-3.5-mini-instruct-Q4_K_L.gguf

⚡ Optional (Recommended)

For better performance and future scalability:

pip install accelerate

✅ Final Setup Verification (Optional)

Run the following commands to verify successful installation:

python -c "import whisper, torch, numpy, sklearn; print('Core OK')"
python -c "from llama_cpp import Llama; print('LLM OK')"
python -c "from sentence_transformers import SentenceTransformer; print('Embeddings OK')"


If all commands run without errors, your environment is ready 🎉

🚀 Ready to Run

You can now execute the pipeline step-by-step:

python step_1_speech_to_text.py
python step_2_topic_segmentation.py
...
python step_7_final_evaluation_and_report.py
