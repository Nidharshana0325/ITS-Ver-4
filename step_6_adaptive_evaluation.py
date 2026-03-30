"""
STEP 6 — Adaptive Evaluation (UPGRADED)
─────────────────────────────────────────
3-level MCQ quiz (easy → medium → difficult).
On wrong answer: teacher gives SHORT spoken explanation + board hint lines.
Uses RAG (ChromaDB) to pull relevant context for feedback.
Pass threshold: 80%.  Retry on fail with re-generated remediation.

Input:  data/mcqs.json, data/processed_topics.json, data/rag_store/
Output: data/quiz_results.json  (consumed by Step 7)
"""

import json
from pathlib import Path
from collections import defaultdict
from llama_cpp import Llama
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR    = Path("data")
MCQ_PATH    = DATA_DIR / "mcqs.json"
TOPICS_PATH = DATA_DIR / "processed_topics.json"
RAG_DIR     = DATA_DIR / "rag_store"
OUTPUT_PATH = DATA_DIR / "quiz_results.json"
MODEL_PATH  = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"
PASS_THRESHOLD = 0.8

# ── load model ────────────────────────────────────────────────────────────────
print("🧠 Loading model...")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8, temperature=0.3, verbose=False)
print("✅ Model ready\n")

# ── load data ─────────────────────────────────────────────────────────────────
with open(MCQ_PATH,    "r", encoding="utf-8") as f: mcqs   = json.load(f)
with open(TOPICS_PATH, "r", encoding="utf-8") as f: topics = json.load(f)

topic_map = {t["topic_id"]: t for t in topics}

# ── RAG ───────────────────────────────────────────────────────────────────────
rag_client    = chromadb.PersistentClient(path=str(RAG_DIR))
embed_fn       = embedding_functions.DefaultEmbeddingFunction()
rag_collection = rag_client.get_or_create_collection(
    name="topics", embedding_function=embed_fn
)

def rag_context(query: str, n=2) -> str:
    try:
        results = rag_collection.query(query_texts=[query], n_results=n)
        return "\n---\n".join(results["documents"][0])
    except Exception:
        return ""

# ── prompts ───────────────────────────────────────────────────────────────────
def wrong_answer_prompt(question, options, chosen, correct, context):
    return f"""<|system|>
You are a friendly teacher giving SHORT constructive feedback.
Keep total response under 3 sentences. Be warm, not harsh.
<|end|>

<|user|>
Question: {question}
Student chose: {chosen}
Correct answer: {correct}
Context: {context}

In 2-3 SHORT sentences: why student's choice was wrong, why correct answer is right.
<|end|>

<|assistant|>
"""

def remediation_prompt(concept, context):
    return f"""<|system|>
You are a tutor re-teaching a concept very simply.
<|end|>

<|user|>
Concept to re-teach: {concept}
Context: {context}

Re-explain in 2-3 simple sentences with one quick example.
<|end|>

<|assistant|>
"""

# ── quiz engine ───────────────────────────────────────────────────────────────
all_results = {}

# We return board_hint + short_explanation as structured data
# so the frontend/API can display them on the board and as speech

def run_level(level: str):
    print(f"\n{'='*60}")
    print(f"🧪  {level.upper()} LEVEL")
    print(f"{'='*60}")

    level_mcqs   = [q for q in mcqs if q["difficulty"] == level]
    level_results = []

    while True:
        correct_count = 0
        weak_concepts = defaultdict(int)

        for idx, q in enumerate(level_mcqs):
            print(f"\nQ{idx+1}: {q['question']}")
            for i, opt in enumerate(q["options"]):
                print(f"  {i}. {opt}")

            try:
                answer = int(input("Your answer (0-3): ").strip())
            except ValueError:
                answer = -1

            is_correct = (answer == q["correct_index"])
            chosen_text = q["options"][answer] if 0 <= answer < 4 else "Invalid"
            correct_text= q["options"][q["correct_index"]]

            result_entry = {
                "question":       q["question"],
                "correct":        is_correct,
                "chosen":         chosen_text,
                "correct_answer": correct_text,
                "board_hint":     q.get("board_hint", ""),
                "short_explanation": q.get("short_explanation", ""),
                "teacher_feedback": ""
            }

            if is_correct:
                print("✅ Correct!")
                correct_count += 1
            else:
                print("❌ Wrong")
                weak_concepts[q["concept"]] += 1

                # RAG-enhanced feedback
                context  = rag_context(q["question"])
                prompt   = wrong_answer_prompt(q["question"], q["options"],
                                               chosen_text, correct_text, context)
                out      = llm(prompt, max_tokens=200)
                feedback = out["choices"][0]["text"].strip()
                result_entry["teacher_feedback"] = feedback

                print(f"\n🎙️  Teacher: {feedback}")
                print(f"📋 Board hint: {q.get('board_hint','')}")

            level_results.append(result_entry)

        score = correct_count / len(level_mcqs) if level_mcqs else 0
        print(f"\n📊 Score: {round(score*100)}%  ({correct_count}/{len(level_mcqs)})")

        if score >= PASS_THRESHOLD:
            print(f"🎉 Passed {level.upper()}!")
            break

        print(f"\n🔁 Below {int(PASS_THRESHOLD*100)}% — remediating weak concepts...")
        for concept in weak_concepts:
            ctx    = rag_context(concept)
            prompt = remediation_prompt(concept, ctx)
            out    = llm(prompt, max_tokens=250)
            print(f"\n📘 Re-teaching '{concept}':")
            print(out["choices"][0]["text"].strip())

        print("\n🔄 Retrying level...\n")

    all_results[level] = level_results

for level in ["easy", "medium", "difficult"]:
    run_level(level)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n✅ Step 6 complete → {OUTPUT_PATH}")
print("🎓 All MCQ levels passed. Proceed to Step 7.")