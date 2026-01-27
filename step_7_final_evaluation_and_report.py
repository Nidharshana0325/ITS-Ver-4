import json
from pathlib import Path
from llama_cpp import Llama

# =========================
# PATHS
# =========================

DATA_DIR = Path("data")
TOPICS_PATH = DATA_DIR / "processed_topics.json"
MCQ_PATH = DATA_DIR / "mcqs.json"
OUTPUT_PATH = DATA_DIR / "final_report.json"

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

print("✅ Model loaded\n")

# =========================
# LOAD DATA
# =========================

with open(TOPICS_PATH, "r", encoding="utf-8") as f:
    topics = json.load(f)

with open(MCQ_PATH, "r", encoding="utf-8") as f:
    mcqs = json.load(f)

# =========================
# BUILD CONTENT SUMMARY
# =========================

all_explanations = "\n".join(
    f"- {t['clean_explanation']}" for t in topics
)

# =========================
# DESCRIPTIVE QUESTIONS
# =========================

descriptive_questions = [
    "Explain the most important concept you learned in this video in your own words.",
    "Choose one concept from the lesson and explain why it is important, using an example."
]

student_answers = []

print("📝 Final Descriptive Evaluation\n")

for q in descriptive_questions:
    print("\n" + "=" * 60)
    print("Question:")
    print(q)
    answer = input("\nYour answer:\n").strip()
    student_answers.append({
        "question": q,
        "answer": answer
    })

# =========================
# EVALUATE DESCRIPTIVE ANSWERS
# =========================

evaluations = []

def evaluation_prompt(question, answer, content):
    return f"""
<|system|>
You are an expert teacher evaluating a student's written answer.
Be constructive and specific.
<|end|>

<|user|>
Question:
{question}

Student answer:
{answer}

Relevant instructional content:
{content}

Evaluate the answer on:
1. Conceptual correctness
2. Completeness
3. Clarity

Then give improvement suggestions.
<|end|>

<|assistant|>
"""

print("\n🧠 Evaluating descriptive answers...\n")

for item in student_answers:
    prompt = evaluation_prompt(
        item["question"],
        item["answer"],
        all_explanations
    )

    output = llm(prompt, max_tokens=500)
    feedback = output["choices"][0]["text"].strip()

    print("Feedback:")
    print(feedback)

    evaluations.append({
        "question": item["question"],
        "student_answer": item["answer"],
        "feedback": feedback
    })

# =========================
# FINAL REPORT GENERATION
# =========================

def report_prompt(topics, mcqs, evaluations):
    topic_list = ", ".join(
        f"Topic {t['topic_id']}" for t in topics
    )

    return f"""
<|system|>
You are an intelligent tutoring system generating a final learning report.
<|end|>

<|user|>
Topics covered:
{topic_list}

MCQ summary:
Total questions: {len(mcqs)}

Descriptive evaluations:
{evaluations}

Generate a final student-facing report with:
- Topics learned
- Strengths
- Weak areas
- Final summary notes
- Study recommendations
<|end|>

<|assistant|>
"""

prompt = report_prompt(topics, mcqs, evaluations)

output = llm(prompt, max_tokens=700)
final_report_text = output["choices"][0]["text"].strip()

# =========================
# SAVE FINAL REPORT
# =========================

final_report = {
    "descriptive_evaluations": evaluations,
    "final_report": final_report_text
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

print("\n✅ STEP 7 COMPLETE")
print(f"📄 Final report saved to: {OUTPUT_PATH}")
