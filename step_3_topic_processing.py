import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "topics.json"
OUTPUT_PATH = DATA_DIR / "processed_topics.json"

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# =========================
# LOAD MODEL
# =========================

print("🧠 Loading Phi-3.5-mini-instruct with llama.cpp (CPU)...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,   # change to number of CPU cores
    temperature=0.3,
    verbose=False
)

print("✅ Model loaded")

# =========================
# PROMPT BUILDER (PHI FORMAT)
# =========================

def build_prompt(text: str) -> str:
    return f"""
<|system|>
You are an expert teacher who explains concepts clearly to beginners.
You never introduce new concepts not present in the lecture.
<|end|>

<|user|>
Explain the following lecture segment in very simple language.

Rules:
- Do NOT add new concepts
- Be concise
- Use exactly ONE simple example
- List 3–5 key points
- Mention prerequisites or write "None"

Lecture segment:
{text}

Return STRICT JSON:
{{
  "clean_explanation": "...",
  "key_points": ["...", "..."],
  "example": "...",
  "prerequisites": ["..."]
}}
<|end|>

<|assistant|>
"""

# =========================
# LOAD TOPICS
# =========================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

processed_topics = []

# =========================
# PROCESS TOPICS
# =========================

for idx, topic in enumerate(topics):
    print(f"🔍 Processing topic {idx + 1}/{len(topics)}")

    raw_text = " ".join(topic["texts"]).strip()
    prompt = build_prompt(raw_text)

    output = llm(prompt, max_tokens=600)
    response = output["choices"][0]["text"]

    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        structured = json.loads(response[start:end])
    except Exception:
        print("⚠️ JSON parse failed, fallback used.")
        structured = {
            "clean_explanation": raw_text[:600],
            "key_points": [],
            "example": "",
            "prerequisites": ["None"]
        }

    processed_topics.append({
        "topic_id": idx,
        "start": topic["start"],
        "end": topic["end"],
        "raw_text": raw_text,
        "clean_explanation": structured["clean_explanation"],
        "key_points": structured["key_points"],
        "example": structured["example"],
        "prerequisites": structured["prerequisites"]
    })

# =========================
# SAVE OUTPUT
# =========================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(processed_topics, f, indent=2, ensure_ascii=False)

print("✅ STEP 3 COMPLETE")
