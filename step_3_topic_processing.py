"""
STEP 3 — Structured Lesson Generation
══════════════════════════════════════════════════════════════════════
Reads segmented topics → calls LLM → writes structured lesson data.

Each topic gets three streams of content:
  1. teacher_notes   — full spoken explanation (150-250 words)
  2. board_notes     — 4 bullet-point key facts (4-8 words each)
  3. diagram         — Mermaid.js diagram (flowchart / mindmap / etc.)

Input:  data/topics.json          (from step_2)
Output: data/processed_topics.json
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List

from llama_cpp import Llama

DATA_DIR    = Path("data")
INPUT_PATH  = DATA_DIR / "topics.json"
OUTPUT_PATH = DATA_DIR / "processed_topics.json"
MODEL_PATH  = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"

# ── Model ────────────────────────────────────────────────────────────
print("🧠 Loading Phi-3.5-mini-instruct…")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096, n_threads=8,
    temperature=0.25, top_p=0.85,
    repeat_penalty=1.1, verbose=False,
)
print("✅ Model ready.\n")


# ── Prompts ──────────────────────────────────────────────────────────

LESSON_PROMPT = """\
<|user|>
You are an expert lesson planner. Create a complete structured lesson.

Topic title: {title}
Topic type:  {topic_type}
Source content:
{content}

Write the lesson using EXACTLY these four section headers, in this order:

TEACHER_NOTES:
Write 150-250 words as a warm, friendly Indian teacher explaining to a 12-year-old.
Use connector phrases: "So basically", "Think of it this way", "Let me give you an example".
NO greetings. Start directly with the explanation.

BOARD_NOTES:
Write exactly 4 bullet points starting with "- ". Each 4-8 words. Make them memorable.
Example:
- Rainwater flows across land surface
- Small streams join into rivers
- Rivers carry water to oceans
- Water cycle repeats continuously

DIAGRAM_TYPE:
Choose one: flowchart_td | flowchart_lr | mindmap | timeline | quadrant | classDiagram | er
- flowchart_td → steps, sequence, process, cycle
- flowchart_lr → cause and effect, input → output
- mindmap      → categories, types, definition, components
- timeline     → history, chronological events
- quadrant     → comparisons, pros/cons
- classDiagram → classification, hierarchy
- er           → entity relationships

DIAGRAM_CODE:
Write valid Mermaid.js code. Rules:
- Each node label MUST be 3-6 words (NO single words like "Hence", "Water", "From")
- Maximum 6 nodes total
- For flowchart_td: use "flowchart TD\n    A[phrase] --> B[phrase]" etc.
- For mindmap: use "mindmap\n  root((\"{title}\"))\n    [\"Concept phrase\"]\n    ..."

<|end|>
<|assistant|>
"""

REMEDIATION_PROMPT = """\
<|user|>
A student is struggling with this concept: {concept}
Context: {context}

Re-explain in 2-3 simple sentences with one quick example.
<|end|>
<|assistant|>
"""


# ── Parsers ──────────────────────────────────────────────────────────

def parse_lesson_response(raw: str, title: str, content: str, topic_type: str) -> Dict:
    """Parse LLM output into structured lesson dict."""
    lesson = {
        "teacher_notes": "",
        "board_notes":   [],
        "diagram_type":  "mindmap",
        "diagram_code":  "",
    }

    # ── teacher_notes ────────────────────────────────────────────────
    tn = re.search(
        r"TEACHER_NOTES:\s*(.*?)(?=\n\s*BOARD_NOTES:|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if tn:
        notes = re.sub(r"\s+", " ", tn.group(1)).strip()
        notes = re.sub(r"^TEACHER_NOTES:?\s*", "", notes, flags=re.IGNORECASE)
        lesson["teacher_notes"] = notes

    # ── board_notes ──────────────────────────────────────────────────
    bn = re.search(
        r"BOARD_NOTES:\s*(.*?)(?=\n\s*DIAGRAM_TYPE:|\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if bn:
        bullets = re.findall(r"[-•*]\s*(.+?)(?=\n[-•*]|\Z)", bn.group(1), re.DOTALL)
        if bullets:
            lesson["board_notes"] = [b.strip() for b in bullets[:4]]
        else:
            for line in bn.group(1).strip().splitlines():
                line = line.strip().strip("-•* ")
                if line and len(line.split()) >= 3:
                    lesson["board_notes"].append(line)
                    if len(lesson["board_notes"]) >= 4:
                        break

    # ── diagram_type ─────────────────────────────────────────────────
    dt = re.search(r"DIAGRAM_TYPE:\s*(\S+)", raw, re.IGNORECASE)
    if dt:
        raw_type = dt.group(1).strip().lower().rstrip(".,;:")
        type_map = {
            "flowchart_td": "flowchart_td",
            "flowchart_lr": "flowchart_lr",
            "flowchart":    "flowchart_td",
            "mindmap":      "mindmap",
            "timeline":     "timeline",
            "quadrant":     "quadrant",
            "classdiagram": "classDiagram",
            "er":           "er",
        }
        lesson["diagram_type"] = type_map.get(raw_type, "mindmap")

    # ── diagram_code ─────────────────────────────────────────────────
    dc = re.search(
        r"DIAGRAM_CODE:\s*(.*?)(?=\Z)",
        raw, re.DOTALL | re.IGNORECASE
    )
    if dc:
        code = dc.group(1).strip()
        code = re.sub(r"^```[a-z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code).strip()
        lesson["diagram_code"] = code

    return lesson


def choose_diagram_type(content: str, topic_type: str) -> str:
    c = content.lower()
    if any(w in c for w in ["process", "step", "stage", "phase", "cycle", "sequence", "procedure"]):
        return "flowchart_td"
    if any(w in c for w in ["cause", "effect", "leads to", "results in", "because", "therefore"]):
        return "flowchart_lr"
    if any(w in c for w in ["type", "kind", "category", "component", "part of", "consist"]):
        return "mindmap"
    if any(w in c for w in ["history", "timeline", "century", "year", "era", "period"]):
        return "timeline"
    if any(w in c for w in ["compare", "contrast", "versus", "vs", "advantage", "disadvantage"]):
        return "quadrant"
    type_map = {
        "definition": "mindmap",
        "process":    "flowchart_td",
        "cause_effect": "flowchart_lr",
        "example":    "mindmap",
        "comparison": "quadrant",
        "properties": "mindmap",
        "general":    "mindmap",
    }
    return type_map.get(topic_type, "mindmap")


def validate_diagram_nodes(code: str) -> str:
    """Replace single-word node labels with meaningful phrases."""
    if not code:
        return code

    BAD_WORDS = {
        "Hence", "Water", "From", "Enters", "Lithosphere", "Runoff",
        "Flow", "Stream", "River", "Ocean", "Evaporation", "Condensation",
        "Precipitation", "Infiltration", "Groundwater", "Sun", "Heat",
        "Air", "Cloud", "Rain", "Land",
    }

    def fix_bracket(m):
        label = m.group(1).strip()
        if len(label.split()) < 2 and label in BAD_WORDS:
            return f"[{label} — key concept here]"
        return m.group(0)

    code = re.sub(r"\[([^\]]+)\]", fix_bracket, code)
    return code


# ── Fallback generators ───────────────────────────────────────────────

def fallback_diagram(title: str, diagram_type: str, concepts: list) -> str:
    """Generate a safe fallback Mermaid diagram."""
    safe = [c[:40] for c in (concepts or ["Key concept", "Important idea", "Real example", "Summary"])]
    while len(safe) < 4:
        safe.append("Related idea here")

    if diagram_type == "flowchart_td":
        return (
            f"flowchart TD\n"
            f"    A[\"{safe[0]}\"] --> B[\"{safe[1]}\"]\n"
            f"    B --> C[\"{safe[2]}\"]\n"
            f"    C --> D[\"{safe[3]}\"]"
        )
    if diagram_type == "flowchart_lr":
        return (
            f"flowchart LR\n"
            f"    A[\"{safe[0]}\"] --> B[\"{safe[1]}\"]\n"
            f"    B --> C[\"{safe[2]}\"]\n"
            f"    C --> D[\"{safe[3]}\"]"
        )
    if diagram_type == "timeline":
        return (
            f"timeline\n"
            f"    title {title}\n"
            f"    section Key Steps\n"
            f"    Step 1 : {safe[0]}\n"
            f"    Step 2 : {safe[1]}\n"
            f"    Step 3 : {safe[2]}"
        )
    # Default: mindmap
    return (
        f'mindmap\n'
        f'  root(("{title}"))\n'
        f'    ["{safe[0]}"]\n'
        f'    ["{safe[1]}"]\n'
        f'    ["{safe[2]}"]\n'
        f'    ["{safe[3]}"]'
    )


def generate_fallback_lesson(title: str, content: str, topic_type: str) -> Dict:
    words    = [w for w in content.split() if len(w) > 4][:6]
    concepts = [" ".join(words[i:i+2]) for i in range(0, min(len(words), 6), 2)][:4]
    if not concepts:
        concepts = ["Key concept here", "Important idea", "Real example", "Summary point"]

    dtype = choose_diagram_type(content, topic_type)
    return {
        "teacher_notes": (
            f"Let's understand {title}. {content[:300]} "
            "Think of it this way — everything we learn connects to real life. "
            f"That's exactly what makes {title} so important to understand."
        ),
        "board_notes": [
            f"{title} — key concept",
            "Connects to real-world examples",
            "Remember the main steps",
            "Practice makes perfect",
        ],
        "diagram_type": dtype,
        "diagram_code": fallback_diagram(title, dtype, concepts),
    }


# ── Title derivation ──────────────────────────────────────────────────

def derive_title(raw_text: str, idx: int) -> str:
    sents = re.split(r"[.!?]+", raw_text)
    for s in sents:
        words = s.strip().split()
        if len(words) >= 4:
            if re.match(r"^(so|well|okay|right|now|hello|hi)\b", s.strip(), re.IGNORECASE):
                continue
            return " ".join(words[:6]).strip(" .,;:!?")
    return f"Topic {idx + 1}"


# ── LLM lesson generation ─────────────────────────────────────────────

def generate_structured_lesson(title: str, topic_type: str, content: str) -> Dict:
    prompt = LESSON_PROMPT.format(
        title=title,
        topic_type=topic_type,
        content=content[:1400],
    )
    try:
        out = llm(prompt, max_tokens=1100, temperature=0.25)
        raw = out["choices"][0]["text"].strip()
        print(f"    LLM preview: {raw[:180]}…")

        lesson = parse_lesson_response(raw, title, content, topic_type)

        # Validate teacher_notes
        if not lesson["teacher_notes"] or len(lesson["teacher_notes"].split()) < 25:
            print("    ⚠️  Teacher notes too short → using fallback")
            return generate_fallback_lesson(title, content, topic_type)

        # Ensure 4 board_notes
        while len(lesson["board_notes"]) < 4:
            lesson["board_notes"].append(f"Understand {title}")
        lesson["board_notes"] = lesson["board_notes"][:4]

        # Validate / fix diagram
        if not lesson["diagram_code"] or len(lesson["diagram_code"]) < 20:
            lesson["diagram_type"] = choose_diagram_type(content, topic_type)
            lesson["diagram_code"] = fallback_diagram(
                title, lesson["diagram_type"],
                [n.split()[:4] and " ".join(n.split()[:4]) for n in lesson["board_notes"]]
            )
        else:
            lesson["diagram_code"] = validate_diagram_nodes(lesson["diagram_code"])

        print("    ✓ Lesson generated successfully")
        return lesson

    except Exception as e:
        print(f"    ⚠️  Error: {str(e)[:60]} → using fallback")
        return generate_fallback_lesson(title, content, topic_type)


# ── Build one processed topic ─────────────────────────────────────────

def process_topic(topic: Dict, idx: int) -> Dict:
    raw_text   = " ".join(topic.get("texts", [])).strip() or f"Topic {idx + 1}."
    title      = derive_title(raw_text, idx)
    topic_type = topic.get("topic_type", "general")

    print(f"\n📚  Topic {idx + 1}: [{topic_type}] {title[:55]}…")
    lesson = generate_structured_lesson(title, topic_type, raw_text[:1800])

    board_colors = ["yellow", "cyan", "lime", "orange"]
    return {
        # Core identity
        "topic_id":      idx,
        "start":         topic.get("start", 0),
        "end":           topic.get("end",   0),
        "topic_title":   title,
        "topic_type":    topic_type,
        "quality_score": topic.get("quality_score", 0.5),
        "is_meaningful": topic.get("is_meaningful", True),

        # Three content streams
        "teacher_notes": lesson["teacher_notes"],
        "board_notes":   lesson["board_notes"],

        # Diagram packaged for frontend
        "diagram": {
            "type":    lesson["diagram_type"],
            "mermaid": lesson["diagram_code"],
        },

        # Aliases for frontend / API compatibility
        "teacher_speech": lesson["teacher_notes"],
        "board_content": [
            {"text": f"→ {pt}", "color": board_colors[i % 4], "indent": 0}
            for i, pt in enumerate(lesson["board_notes"])
        ],

        # RAG / MCQ helpers
        "key_points":    lesson["board_notes"][:3],
        "texts":         topic.get("texts", []),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("\n=== STEP 3: Structured Lesson Generation ===\n")

    if not INPUT_PATH.exists():
        print(f"❌ {INPUT_PATH} not found — run step_2 first.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if not topics:
        print("⚠️  No topics found.")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return

    print(f"📂 Loaded {len(topics)} segmented topics\n")
    processed = []

    for idx, topic in enumerate(topics):
        try:
            result = process_topic(topic, idx)
            processed.append(result)
            time.sleep(0.2)
        except Exception as err:
            print(f"❌  Critical error on topic {idx}: {err}")
            fallback = generate_fallback_lesson(
                f"Topic {idx + 1}",
                " ".join(topic.get("texts", [])),
                topic.get("topic_type", "general"),
            )
            processed.append({
                "topic_id":      idx,
                "start":         topic.get("start", 0),
                "end":           topic.get("end",   0),
                "topic_title":   f"Topic {idx + 1}",
                "topic_type":    topic.get("topic_type", "general"),
                "quality_score": 0.0,
                "is_meaningful": False,
                "teacher_notes": fallback["teacher_notes"],
                "board_notes":   fallback["board_notes"],
                "diagram":       {"type": fallback["diagram_type"], "mermaid": fallback["diagram_code"]},
                "teacher_speech": fallback["teacher_notes"],
                "board_content": [
                    {"text": f"→ {p}", "color": c, "indent": 0}
                    for p, c in zip(fallback["board_notes"], ["yellow","cyan","lime","orange"])
                ],
                "key_points": fallback["board_notes"][:3],
                "texts":      topic.get("texts", []),
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"\n{'═'*60}")
    print(f"✅  STEP 3 COMPLETE — {len(processed)} topics → {OUTPUT_PATH}")
    print(f"{'═'*60}\n")
    for p in processed:
        q = p.get("quality_score", 0)
        stars = "⭐⭐⭐" if q > 0.7 else "⭐⭐" if q > 0.5 else "⭐"
        print(f"  {stars}  {p['topic_title'][:50]:<50}  q={q:.2f}")
    print(f"\n   Next: run step_4_rag_builder.py\n")


if __name__ == "__main__":
    main()