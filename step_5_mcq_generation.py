"""
STEP 5 — MCQ Generation
══════════════════════════════════════════════════════════════════════
Generates MCQs from processed topic content using the Phi-3.5 model.

Per-topic MCQs:
  - 1 easy   (direct recall / definition)
  - 1 medium (application / connecting concepts)
  - 1 hard   (inference / multi-step)

  → Total = N_topics × 3 questions

Plus 3 bonus "cross-topic" questions across all content.

Input:  data/processed_topics.json
Output: data/mcqs.json

Run after step_4_rag_builder.py, before launching the server.
"""

import json
import re
from pathlib import Path

from llama_cpp import Llama

DATA_DIR   = Path("data")
INPUT      = DATA_DIR / "processed_topics.json"
OUTPUT     = DATA_DIR / "mcqs.json"
MODEL_PATH = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# ── Difficulty descriptors ─────────────────────────────────────────
DIFFICULTY_SPEC = {
    "easy": (
        "direct recall or definition — student just needs to remember what was taught",
        "Bloom's Level 1–2 (Remember/Understand)"
    ),
    "medium": (
        "application or connecting two concepts — student must use knowledge",
        "Bloom's Level 3–4 (Apply/Analyse)"
    ),
    "difficult": (
        "inference, multi-step reasoning, or real-world evaluation — requires higher-order thinking",
        "Bloom's Level 5–6 (Evaluate/Create)"
    ),
}

MCQ_PROMPT = """<|system|>
You are an expert assessment designer who writes clear, unambiguous MCQs
that test GENUINE understanding, not word-matching.
<|end|>
<|user|>
Topic: {title}
Difficulty: {difficulty} — {spec} ({bloom})

Content:
{content}

Write ONE MCQ for this topic at the specified difficulty.

Rules:
- Question must be clear and answerable from the content
- All 4 options must be plausible (no obvious throw-aways)
- correct_index is 0-based index of the correct option
- board_hint: ≤ 6 words, shown on chalkboard when student is wrong
- short_explanation: 1-2 sentences explaining WHY the answer is correct

Return ONLY a valid JSON object — no markdown, no extra text:
{{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_index": 0,
  "difficulty": "{difficulty}",
  "concept": "{title}",
  "board_hint": "short hint ≤6 words",
  "short_explanation": "Why this answer is correct."
}}
<|end|>
<|assistant|>
"""

CROSS_TOPIC_PROMPT = """<|system|>
You are an expert assessment designer. Write a cross-topic MCQ that
connects ideas from multiple topics covered in a lesson.
<|end|>
<|user|>
Topics covered in this lesson:
{topics_list}

Full content summary:
{summary}

Write ONE {difficulty} MCQ that connects at least 2 of these topics.

Rules:
- Question tests understanding across topics, not just one isolated fact
- All 4 options plausible
- correct_index is 0-based

Return ONLY valid JSON — no markdown, no extra text:
{{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_index": 0,
  "difficulty": "{difficulty}",
  "concept": "Cross-topic",
  "board_hint": "short hint ≤6 words",
  "short_explanation": "Why this answer is correct."
}}
<|end|>
<|assistant|>
"""


def extract_json_object(raw: str) -> dict | None:
    """Robustly extract the first JSON object from raw LLM output."""
    # Strip markdown code fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.rstrip())
    s, e = raw.find("{"), raw.rfind("}") + 1
    if s == -1 or e == 0:
        return None
    try:
        return json.loads(raw[s:e])
    except Exception:
        return None


def validate_mcq(q: dict) -> bool:
    """Check required fields and sensible values."""
    required = ["question", "options", "correct_index", "difficulty"]
    if not all(k in q for k in required):
        return False
    if not isinstance(q["options"], list) or len(q["options"]) < 2:
        return False
    idx = q["correct_index"]
    if not isinstance(idx, int) or idx < 0 or idx >= len(q["options"]):
        return False
    if len(q["question"].strip()) < 10:
        return False
    return True


def generate_mcq_for_topic(
    llm: Llama,
    topic: dict,
    difficulty: str,
    attempt: int = 0,
) -> dict | None:
    """Generate one MCQ for a topic at a given difficulty. Retries once."""
    title   = topic.get("topic_title", f"Topic {topic['topic_id']+1}")
    notes   = topic.get("teacher_notes") or topic.get("teacher_speech") or ""
    board   = " | ".join(topic.get("board_notes", []))
    content = f"{notes[:800]}\nKey points: {board}"

    spec, bloom = DIFFICULTY_SPEC[difficulty]
    prompt = MCQ_PROMPT.format(
        title=title, difficulty=difficulty,
        spec=spec, bloom=bloom, content=content
    )

    try:
        out = llm(prompt, max_tokens=400)
        raw = out["choices"][0]["text"].strip()
        q   = extract_json_object(raw)
        if q and validate_mcq(q):
            q["difficulty"] = difficulty  # enforce correct label
            return q
    except Exception as e:
        print(f"      ⚠️  LLM error: {e}")

    # Retry once with simplified prompt
    if attempt == 0:
        return generate_mcq_for_topic(llm, topic, difficulty, attempt=1)

    return None


def generate_cross_topic_mcqs(
    llm: Llama,
    topics: list,
    count: int = 3,
) -> list:
    """Generate cross-topic MCQs connecting multiple topics."""
    titles_list = "\n".join(f"- {t.get('topic_title','?')}" for t in topics)
    summary_parts = []
    for t in topics:
        notes = (t.get("teacher_notes") or t.get("teacher_speech") or "")[:150]
        summary_parts.append(f"[{t.get('topic_title','')}]: {notes}")
    summary = "\n".join(summary_parts)[:1800]

    difficulties = ["easy", "medium", "difficult"]
    results = []

    for i in range(count):
        diff = difficulties[i % len(difficulties)]
        prompt = CROSS_TOPIC_PROMPT.format(
            topics_list=titles_list,
            summary=summary,
            difficulty=diff,
        )
        try:
            out = llm(prompt, max_tokens=400)
            raw = out["choices"][0]["text"].strip()
            q   = extract_json_object(raw)
            if q and validate_mcq(q):
                q["difficulty"] = diff
                q["concept"]    = "Cross-topic"
                results.append(q)
                print(f"   ✓ Cross-topic {diff} MCQ generated")
            else:
                print(f"   ⚠️  Cross-topic {diff} parse failed — skipping")
        except Exception as e:
            print(f"   ⚠️  Cross-topic error: {e}")

    return results


def main():
    print("\n=== STEP 5: MCQ Generation ===\n")

    if not INPUT.exists():
        print(f"❌ {INPUT} not found — run step_3 first.")
        return

    with open(INPUT, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if not topics:
        print("⚠️  No topics found.")
        return

    print(f"🧠 Loading Phi-3.5-mini-instruct…")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048, n_threads=8,
        temperature=0.3, verbose=False,
    )
    print(f"✅ Model ready. Generating MCQs for {len(topics)} topics…\n")

    all_mcqs = []
    difficulties = ["easy", "medium", "difficult"]

    for topic in topics:
        title = topic.get("topic_title", f"Topic {topic['topic_id']+1}")
        print(f"\n📚 Topic: {title[:50]}")
        for diff in difficulties:
            q = generate_mcq_for_topic(llm, topic, diff)
            if q:
                all_mcqs.append(q)
                print(f"   ✓ [{diff:10}] {q['question'][:60]}…")
            else:
                print(f"   ✗ [{diff:10}] Failed — skipping")

    # Cross-topic MCQs (optional, skip if only 1 topic)
    if len(topics) >= 2:
        print("\n🔗 Generating cross-topic MCQs…")
        cross = generate_cross_topic_mcqs(llm, topics, count=min(3, len(topics)))
        all_mcqs.extend(cross)

    # Save
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_mcqs, f, indent=2, ensure_ascii=False)

    counts = {d: sum(1 for q in all_mcqs if q.get("difficulty") == d) for d in difficulties}
    print(f"\n✅ Step 5 complete — {len(all_mcqs)} MCQs saved → {OUTPUT}")
    print(f"   Easy: {counts['easy']} | Medium: {counts['medium']} | Difficult: {counts['difficult']}")
    print("   Run the server next: uvicorn app:app --reload --port 8000\n")


if __name__ == "__main__":
    main()