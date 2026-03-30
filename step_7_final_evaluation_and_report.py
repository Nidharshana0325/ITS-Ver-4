"""
STEP 7 — Final Evaluation & Report
────────────────────────────────────
Descriptive Q&A → LLM evaluation → final JSON report.
Uses RAG for richer evaluation context.

Input:  data/processed_topics.json, data/mcqs.json, data/quiz_results.json
Output: data/final_report.json
"""

import json
from pathlib import Path
from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR     = Path("data")
TOPICS_PATH  = DATA_DIR / "processed_topics.json"
MCQ_PATH     = DATA_DIR / "mcqs.json"
QUIZ_PATH    = DATA_DIR / "quiz_results.json"
RAG_DIR      = DATA_DIR / "rag_store"
OUTPUT_PATH  = DATA_DIR / "final_report.json"
MODEL_PATH   = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

print("🧠 Loading model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8, temperature=0.3, verbose=False)
print("✅ Ready\n")

with open(TOPICS_PATH, "r", encoding="utf-8") as f: topics = json.load(f)
with open(MCQ_PATH,    "r", encoding="utf-8") as f: mcqs   = json.load(f)

quiz_results = {}
if QUIZ_PATH.exists():
    with open(QUIZ_PATH, "r", encoding="utf-8") as f:
        quiz_results = json.load(f)

# RAG
rag_client    = chromadb.PersistentClient(path=str(RAG_DIR))
embed_fn       = embedding_functions.DefaultEmbeddingFunction()
rag_collection = rag_client.get_or_create_collection(
    name="topics", embedding_function=embed_fn
)

def rag_context(query, n=3):
    try:
        r = rag_collection.query(query_texts=[query], n_results=n)
        return "\n---\n".join(r["documents"][0])
    except Exception:
        return ""

all_explanations = "\n".join(f"- {t['teacher_speech'][:300]}" for t in topics)

descriptive_questions = [
    "In your own words, explain the most important concept you learned. Give an example.",
    "Pick one topic from the lesson. Explain why it matters and how it connects to something in real life."
]

student_answers = []
print("📝 Final Descriptive Evaluation\n")

for q in descriptive_questions:
    print("\n" + "="*60)
    print(f"Question: {q}")
    answer = input("\nYour answer:\n").strip()
    student_answers.append({"question": q, "answer": answer})

evaluations = []

EVAL_PROMPT = """<|system|>
You are an expert teacher evaluating a student's written answer. Be encouraging.
<|end|>

<|user|>
Question: {question}
Student answer: {answer}
Relevant content: {context}

Score 1-10 and give:
1. What was good
2. What to improve
3. One specific suggestion
Keep total under 100 words.
<|end|>

<|assistant|>
"""

print("\n🧠 Evaluating answers...\n")
for item in student_answers:
    ctx    = rag_context(item["question"])
    prompt = EVAL_PROMPT.format(
        question=item["question"], answer=item["answer"], context=ctx[:800]
    )
    out      = llm(prompt, max_tokens=300)
    feedback = out["choices"][0]["text"].strip()

    print(f"Feedback:\n{feedback}\n")
    evaluations.append({
        "question":       item["question"],
        "student_answer": item["answer"],
        "feedback":       feedback
    })

# Compute quiz summary
quiz_summary = {}
for level, results in quiz_results.items():
    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    quiz_summary[level] = {"total": total, "correct": correct,
                           "pct": round(correct/total*100 if total else 0)}

weak_topics = []
for level, results in quiz_results.items():
    for r in results:
        if not r["correct"]:
            weak_topics.append(r["question"][:80])

REPORT_PROMPT = f"""<|system|>
You are an intelligent tutoring system generating a student learning report.
<|end|>

<|user|>
Topics covered: {[t['topic_title'] for t in topics]}
Quiz summary: {quiz_summary}
Weak areas (wrong questions): {weak_topics[:5]}

Generate a friendly student report (under 150 words) with:
- Topics learned
- Quiz performance summary  
- Top 2 strengths
- Top 2 areas to improve
- 2 study recommendations
<|end|>

<|assistant|>
"""

out         = llm(REPORT_PROMPT, max_tokens=400)
report_text = out["choices"][0]["text"].strip()

final = {
    "topics":                [t["topic_title"] for t in topics],
    "quiz_summary":          quiz_summary,
    "descriptive_evaluations": evaluations,
    "weak_topics":           list(set(weak_topics)),
    "final_report":          report_text
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, ensure_ascii=False)

print(f"✅ Step 7 complete → {OUTPUT_PATH}")