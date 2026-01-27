import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
TOPICS_PATH = DATA_DIR / "processed_topics.json"

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# =========================
# LOAD MODEL
# =========================

print("🧠 Loading Phi-3.5-mini-instruct...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,   # adjust to your CPU
    temperature=0.3,
    verbose=False
)

print("✅ Model loaded\n")

# =========================
# LOAD TOPICS
# =========================

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

# =========================
# PROMPT BUILDER
# =========================

def build_doubt_prompt(context, question):
    return f"""
<|system|>
You are a patient tutor helping a student understand a topic.
You must ONLY use the provided explanation.
You must NOT introduce new concepts.
Explain simply with one example.
<|end|>

<|user|>
Context so far:
{context}

Student question:
{question}

Answer clearly and simply.
<|end|>

<|assistant|>
"""

# =========================
# INTERACTIVE LOOP
# =========================

print("🎓 Intelligent Video Tutor Started")
print("Type 'exit' anytime to quit.\n")

accumulated_context = ""

for topic in topics:
    print("=" * 60)
    print(f"▶ Playing video segment: {topic['start']}s → {topic['end']}s")
    print("📘 Topic explanation:")
    print(topic["clean_explanation"])
    print("\n⏸️  Video paused.")

    # Add topic explanation to context
    accumulated_context += "\n" + topic["clean_explanation"]

    while True:
        user_input = input("\n❓ Do you have any doubts? (yes/no): ").strip().lower()

        if user_input == "exit":
            print("👋 Exiting tutor.")
            exit()

        if user_input == "no":
            print("✅ Moving to next topic...\n")
            break

        if user_input == "yes":
            question = input("✍️ Ask your question: ").strip()

            if question.lower() == "exit":
                print("👋 Exiting tutor.")
                exit()

            prompt = build_doubt_prompt(accumulated_context, question)

            output = llm(prompt, max_tokens=400)
            answer = output["choices"][0]["text"]

            print("\n🧠 Tutor Answer:")
            print(answer.strip())

        else:
            print("⚠️ Please type 'yes' or 'no'.")

print("\n🎉 Video completed! Moving to assessment phase next.")
