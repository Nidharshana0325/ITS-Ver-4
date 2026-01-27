import json
from pathlib import Path
from collections import defaultdict
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
MCQ_PATH = DATA_DIR / "mcqs.json"
TOPICS_PATH = DATA_DIR / "processed_topics.json"

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

PASS_THRESHOLD = 0.8

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

print("✅ Model loaded\n")

# =========================
# LOAD DATA
# =========================

with open(MCQ_PATH, "r", encoding="utf-8") as f:
    mcqs = json.load(f)

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

# Build concept → explanation map
concept_explanations = {
    t["topic_id"]: t["clean_explanation"] for t in topics
}

# =========================
# PROMPT BUILDERS
# =========================

def feedback_prompt(question, options, chosen, correct, explanation):
    return f"""
<|system|>
You are a helpful tutor giving constructive feedback.
Explain mistakes clearly without judgment.
<|end|>

<|user|>
Question:
{question}

Options:
{options}

Student chose: {chosen}
Correct answer: {correct}

Relevant explanation:
{explanation}

Explain:
1. Why the chosen option is wrong
2. Why the correct option is right
<|end|>

<|assistant|>
"""

def remediation_prompt(concept, explanation):
    return f"""
<|system|>
You are a tutor helping a student fix a misconception.
<|end|>

<|user|>
The student is weak in the concept: {concept}

Explain this concept again very simply.
Use one example.
Keep it short.

Explanation:
{explanation}
<|end|>

<|assistant|>
"""

# =========================
# EVALUATION ENGINE
# =========================

def run_level(level):
    print(f"\n🧪 Starting {level.upper()} level")

    level_mcqs = [q for q in mcqs if q["difficulty"] == level]

    while True:
        correct_count = 0
        weak_concepts = defaultdict(int)

        for idx, q in enumerate(level_mcqs):
            print("\n" + "-" * 60)
            print(f"Q{idx + 1}: {q['question']}")

            for i, opt in enumerate(q["options"]):
                print(f"  {i}. {opt}")

            try:
                answer = int(input("Your answer (0-3): ").strip())
            except ValueError:
                print("⚠️ Invalid input. Treated as wrong.")
                answer = -1

            if answer == q["correct_index"]:
                print("✅ Correct")
                correct_count += 1
            else:
                print("❌ Incorrect")
                weak_concepts[q["concept"]] += 1

                explanation = concept_explanations.get(
                    q["concept"], ""
                )

                prompt = feedback_prompt(
                    q["question"],
                    q["options"],
                    q["options"][answer] if 0 <= answer < 4 else "Invalid",
                    q["options"][q["correct_index"]],
                    explanation
                )

                output = llm(prompt, max_tokens=400)
                print("\n🧠 Feedback:")
                print(output["choices"][0]["text"].strip())

        score = correct_count / len(level_mcqs)
        print(f"\n📊 Score: {round(score * 100)}%")

        if score >= PASS_THRESHOLD:
            print(f"🎉 Passed {level.upper()} level!")
            break

        print("\n🔁 Remediation required")

        for concept in weak_concepts:
            explanation = concept_explanations.get(concept, "")
            prompt = remediation_prompt(concept, explanation)
            output = llm(prompt, max_tokens=300)

            print("\n📘 Remedial Explanation:")
            print(output["choices"][0]["text"].strip())

        print("\n🔄 Retrying this level...\n")

# =========================
# RUN ALL LEVELS
# =========================

for difficulty in ["easy", "medium", "difficult"]:
    run_level(difficulty)

print("\n🎓 All levels completed. Proceeding to final evaluation next.")
