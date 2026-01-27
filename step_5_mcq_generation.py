import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "processed_topics.json"
OUTPUT_PATH = DATA_DIR / "mcqs.json"

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# =========================
# LOAD MODEL
# =========================

print("🧠 Loading Phi-3.5-mini-instruct...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    temperature=0.3,
    verbose=False
)

print("✅ Model loaded")

# =========================
# LOAD CONTENT
# =========================

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

# Build full learning context
full_context = "\n".join(
    f"Topic {t['topic_id']}: {t['clean_explanation']}"
    for t in topics
)

# =========================
# PROMPT BUILDER
# =========================

def build_mcq_prompt(context):
    return f"""
<|system|>
You are an expert educational assessment designer.

You generate multiple-choice questions based on instructional content.
You strictly follow cognitive difficulty definitions.
<|end|>

<|user|>
Instructional content:
{context}

Task:
Generate MCQs at three difficulty levels.

Difficulty definitions:
Easy:
- Recall or recognition
- Explicitly stated facts
- Low conceptual complexity

Medium:
- Application of ideas
- Connecting concepts
- Procedures demonstrated in content

Difficult:
- High abstraction
- Inference or reasoning
- Transfer to unfamiliar situations
- Multi-step reasoning not explicitly modeled

Rules:
- 3 Easy MCQs
- 3 Medium MCQs
- 3 Difficult MCQs
- Each MCQ has 4 options
- Only one correct option
- Do NOT add external knowledge

Return STRICT JSON ONLY in this format:
[
  {{
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_index": 0,
    "difficulty": "easy",
    "concept": "..."
  }}
]
<|end|>

<|assistant|>
"""

# =========================
# GENERATE MCQs
# =========================

print("📝 Generating MCQs...")

prompt = build_mcq_prompt(full_context)

output = llm(prompt, max_tokens=1200)
response = output["choices"][0]["text"]

# =========================
# SAFE JSON PARSING
# =========================

try:
    start = response.find("[")
    end = response.rfind("]") + 1
    mcqs = json.loads(response[start:end])
except Exception as e:
    raise RuntimeError("❌ Failed to parse MCQs JSON") from e

# =========================
# SAVE OUTPUT
# =========================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(mcqs, f, indent=2, ensure_ascii=False)

print("✅ STEP 5 COMPLETE")
print(f"📄 MCQs saved to: {OUTPUT_PATH}")
print(f"📊 Total questions generated: {len(mcqs)}")
