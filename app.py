"""
app.py — EduAI Classroom Server
════════════════════════════════════════════════════════════════════════
Endpoints:
  GET  /                        → classroom HTML (from templates/)
  GET  /api/topics              → topic list (for sidebar)
  GET  /api/topic/{id}          → full topic data
  POST /api/ask                 → streaming doubt answer (SSE)
  GET  /api/mcqs/{difficulty}   → MCQs (correct_index hidden from client)
  POST /api/check_answer        → grade MCQ answer + teacher feedback
  POST /api/descriptive         → evaluate written answer (4 criteria)
  GET  /api/report              → final learning report JSON
  GET  /api/quiz_results        → saved quiz results
  GET  /api/health              → server status

Run: uvicorn app:app --reload --port 8000
"""

import json
import asyncio
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
MODEL_PATH   = "models/Phi-3.5-mini-instruct-Q4_K_L.gguf"
TEMPLATE_DIR = Path("templates")
STATIC_DIR   = Path("static")

app = FastAPI(title="EduAI Classroom")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

print("🧠 Loading Phi-3.5-mini-instruct…")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    temperature=0.3,
    verbose=False,
)
print("✅ Model ready\n")


# ── RAG (optional — silently skips if store missing) ───────────────────────────
def rag_context(query: str, n: int = 2) -> str:
    """Retrieve relevant context from ChromaDB RAG store, or return empty string."""
    rag_dir = DATA_DIR / "rag_store"
    if not rag_dir.exists():
        return ""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        client     = chromadb.PersistentClient(path=str(rag_dir))
        embed_fn   = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_or_create_collection("topics", embedding_function=embed_fn)
        results    = collection.query(query_texts=[query], n_results=n)
        docs       = results.get("documents", [[]])[0]
        return "\n---\n".join(docs)
    except Exception:
        return ""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_topic(topic_id: int):
    data = load_json(DATA_DIR / "processed_topics.json") or []
    return next((t for t in data if t["topic_id"] == topic_id), None)


def phi_prompt(system: str, user: str) -> str:
    """Phi-3.5-mini-instruct chat format."""
    return f"<|system|>\n{system}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n"


def clean_board_items(items: list) -> list:
    """
    Normalise board items from LLM output.
    Accepts:
      - list of strings  → wraps into dicts
      - list of dicts    → normalises color/text fields
    Returns list of dicts: [{text, color, indent}]
    """
    COLORS  = ["yellow", "cyan", "lime", "orange", "white", "pink"]
    PALETTE = set(COLORS)
    result  = []

    for i, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item, "color": COLORS[i % len(COLORS)], "indent": 0}

        text = item.get("text", "").strip()
        # Normalise bullet prefixes
        if text.startswith("* "):
            text = "★ " + text[2:]
        elif text.startswith("- "):
            text = "→ " + text[2:]
        item["text"]  = text
        item["color"] = item.get("color") if item.get("color") in PALETTE else COLORS[i % len(COLORS)]
        item["indent"] = int(item.get("indent", 0))
        result.append(item)

    return result


def build_topic_context(topic) -> str:
    """Build a concise context string from topic data for LLM prompts."""
    if not topic:
        return ""
    key_pts = "; ".join(
        topic.get("key_points", topic.get("board_notes", []))[:5]
    )
    notes = (topic.get("teacher_notes") or topic.get("teacher_speech") or "")[:600]
    return (
        f"Topic: {topic['topic_title']}\n"
        f"Teacher explanation: {notes}\n"
        f"Key points: {key_pts}"
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = TEMPLATE_DIR / "classroom.html"
    if not html_path.exists():
        raise HTTPException(500, "classroom.html not found in templates/")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/quiz", response_class=HTMLResponse)
def quiz_page():
    html_path = TEMPLATE_DIR / "quiz.html"
    if not html_path.exists():
        raise HTTPException(500, "quiz.html not found in templates/")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/report", response_class=HTMLResponse)
def report_page():
    html_path = TEMPLATE_DIR / "report.html"
    if not html_path.exists():
        raise HTTPException(500, "report.html not found in templates/")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/topics")
def api_topics():
    data = load_json(DATA_DIR / "processed_topics.json")
    if data is None:
        raise HTTPException(503, "processed_topics.json not found — run step_3 first")
    return JSONResponse([
        {
            "topic_id":      t["topic_id"],
            "topic_title":   t["topic_title"],
            "start":         t["start"],
            "end":           t["end"],
            "duration":      round(t["end"] - t["start"], 1),
            "quality_score": t.get("quality_score", 0.5),
        }
        for t in data
    ])


@app.get("/api/topic/{topic_id}")
def api_topic(topic_id: int):
    t = get_topic(topic_id)
    if t is None:
        raise HTTPException(404, f"Topic {topic_id} not found")
    return JSONResponse(t)


# ── /api/ask — streaming SSE doubt answering ───────────────────────────────────

class AskRequest(BaseModel):
    question: str
    topic_id: int = 0
    history:  list = []


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    topic       = get_topic(req.topic_id)
    topic_ctx   = build_topic_context(topic)

    # Supplement with RAG if available
    rag_ctx = rag_context(req.question)
    if rag_ctx:
        topic_ctx += f"\n\nAdditional context:\n{rag_ctx[:400]}"

    # ── Step 1: Board key-points ─────────────────────────────────────────────
    board_system = (
        "You are an Indian school teacher writing quick chalkboard notes to answer a student's doubt.\n"
        "Return ONLY a valid JSON array of 4–5 items. No markdown, no extra text before or after the JSON.\n"
        'Schema: [{"text":"…","color":"…","indent":0}]\n'
        "Rules:\n"
        "  Item 1: 3–5 word title summarising the question, color=yellow\n"
        "  Items 2–4: key answer points (max 7 words each), start with '→ ', colors: cyan, lime, orange\n"
        "  Item 5 (optional): a simple everyday example, color=white\n"
        "Every item must be meaningful — no 'See explanation' placeholders."
    )
    board_user = (
        f"Lesson context:\n{topic_ctx[:350]}\n\n"
        f"Student's question: {req.question}"
    )
    board_raw = llm(phi_prompt(board_system, board_user), max_tokens=300)["choices"][0]["text"].strip()

    board_items: list = []
    try:
        s, e = board_raw.find("["), board_raw.rfind("]") + 1
        if s >= 0 and e > s:
            board_items = clean_board_items(json.loads(board_raw[s:e]))
    except Exception:
        pass

    if not board_items:
        # Minimal fallback — still meaningful
        short_q = req.question[:38].rstrip()
        board_items = clean_board_items([
            {"text": short_q,           "color": "yellow", "indent": 0},
            {"text": "→ Key concept",   "color": "cyan",   "indent": 0},
            {"text": "→ How it works",  "color": "lime",   "indent": 0},
            {"text": "→ Remember this", "color": "orange", "indent": 0},
        ])

    # ── Step 2: Concept diagram for the doubt ────────────────────────────────
    diag_system = (
        "Generate a simple horizontal Mermaid flowchart (4–5 nodes) to explain a student's question.\n"
        'Return ONLY JSON (no markdown): {"type":"flowchart_lr","mermaid":"…"}\n'
        "Rules:\n"
        "  - Use 'flowchart LR' (horizontal left-to-right)\n"
        "  - Each node: 2–5 meaningful words in square brackets, e.g. A[\"Concept name\"]\n"
        "  - Nodes must form a logical sequence or hierarchy for the topic\n"
        "  - No single-word nodes"
    )
    diag_user = (
        f"Context: {topic_ctx[:250]}\n"
        f"Question: {req.question}"
    )
    diag_raw = llm(phi_prompt(diag_system, diag_user), max_tokens=350)["choices"][0]["text"].strip()

    doubt_diagram: dict = {}
    try:
        # Strip markdown fences if present
        diag_raw = re.sub(r"^```[a-z]*\n?", "", diag_raw, flags=re.I)
        diag_raw = re.sub(r"\n?```$", "", diag_raw.rstrip())
        s, e = diag_raw.find("{"), diag_raw.rfind("}") + 1
        if s >= 0 and e > s:
            parsed = json.loads(diag_raw[s:e])
            if parsed.get("mermaid", "").strip():
                doubt_diagram = parsed
    except Exception:
        pass

    if not doubt_diagram:
        # Fallback: build from board items
        nodes  = "\n".join(f'  N{i}["{it["text"].lstrip("→ ★").strip()[:30]}"]'
                           for i, it in enumerate(board_items[:4]))
        arrows = "\n".join(f"  N{i} --> N{i+1}" for i in range(min(len(board_items)-1, 3)))
        doubt_diagram = {
            "type": "flowchart_lr",
            "mermaid": f"flowchart LR\n{nodes}\n{arrows}",
        }

    # ── Step 3: Spoken answer (streaming) ────────────────────────────────────
    history_str = ""
    for m in req.history[-4:]:
        role = "Student" if m.get("role") == "user" else "Lekha"
        history_str += f"{role}: {m.get('content', '')}\n"

    speech_system = (
        "You are Lekha, a warm, friendly Indian school teacher answering a student's doubt.\n"
        "Rules:\n"
        "  - 3–4 complete sentences only — be concise and clear\n"
        "  - Use simple conversational Indian English (not formal)\n"
        "  - Include one short everyday Indian example (chai, cricket, market, etc.)\n"
        "  - Natural openers: 'See basically,', 'Think of it this way —', 'Good question!'\n"
        "  - End with an encouraging closing line\n"
        "  - Do NOT start with 'I', 'Sure,', 'Hello', 'Certainly', or 'Of course'\n"
        "  - Every sentence must explain something — no filler phrases"
    )
    speech_user = (
        f"Lesson context:\n{topic_ctx[:450]}\n\n"
        f"{history_str}"
        f"Student asks: {req.question}"
    )

    async def event_stream():
        # Event 1: board items + diagram (sent immediately, before streaming answer)
        yield f"data: {json.dumps({'type': 'board', 'items': board_items, 'diagram': doubt_diagram})}\n\n"

        # Events 2…N: stream the spoken answer token by token
        full = ""
        for chunk in llm(phi_prompt(speech_system, speech_user), max_tokens=260, stream=True):
            tok   = chunk["choices"][0]["text"]
            full += tok
            yield f"data: {json.dumps({'type': 'token', 'token': tok})}\n\n"
            await asyncio.sleep(0)  # yield control to event loop

        # Final event: complete answer
        yield f"data: {json.dumps({'type': 'done', 'full': full.strip()})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── MCQs ───────────────────────────────────────────────────────────────────────

@app.get("/api/mcqs/{difficulty}")
def api_mcqs(difficulty: str):
    data = load_json(DATA_DIR / "mcqs.json")
    if data is None:
        raise HTTPException(503, "mcqs.json not found — run step_5 first")
    # Never send correct_index to client
    return JSONResponse([
        {k: v for k, v in q.items() if k != "correct_index"}
        for q in data
        if q.get("difficulty") == difficulty
    ])


class AnswerReq(BaseModel):
    question_index: int
    difficulty:     str
    chosen_index:   int


@app.post("/api/check_answer")
async def api_check_answer(req: AnswerReq):
    data = load_json(DATA_DIR / "mcqs.json")
    if data is None:
        raise HTTPException(503, "mcqs.json not found")

    qs = [q for q in data if q.get("difficulty") == req.difficulty]
    if req.question_index >= len(qs):
        raise HTTPException(404, "Question index out of range")

    q  = qs[req.question_index]
    ok = req.chosen_index == q["correct_index"]

    result = {
        "correct":           ok,
        "correct_index":     q["correct_index"],
        "correct_text":      q["options"][q["correct_index"]],
        "board_hint":        q.get("board_hint", ""),
        "short_explanation": q.get("short_explanation", ""),
        "teacher_feedback":  "",
    }

    if not ok:
        ctx = rag_context(q["question"])
        fb_prompt = phi_prompt(
            system=(
                "You are a kind, encouraging Indian school teacher.\n"
                "Write 2 warm, brief sentences:\n"
                "  1. Why the student's answer was incorrect\n"
                "  2. What the correct answer means in simple terms\n"
                "Use friendly Indian English — like talking to a student, not writing an essay."
            ),
            user=(
                f"Question: {q['question']}\n"
                f"Student chose: {q['options'][req.chosen_index]}\n"
                f"Correct answer: {q['options'][q['correct_index']]}\n"
                f"Context: {ctx[:300]}"
            ),
        )
        result["teacher_feedback"] = llm(fb_prompt, max_tokens=140)["choices"][0]["text"].strip()

    return JSONResponse(result)


# ── Descriptive evaluation ─────────────────────────────────────────────────────

class DescReq(BaseModel):
    question: str
    answer:   str


@app.post("/api/descriptive")
def api_descriptive(req: DescReq):
    ctx = rag_context(req.question)
    eval_prompt = phi_prompt(
        system=(
            "You are an expert teacher evaluating a student's written answer.\n"
            "Evaluate on 4 criteria (1 sentence each):\n"
            "  1. Completeness — did they cover the key points?\n"
            "  2. Correctness  — is the information accurate?\n"
            "  3. Clarity      — is it well explained?\n"
            "  4. HOTS         — do they connect it to real life or broader ideas?\n\n"
            "End with: overall score out of 10, one strength, one improvement.\n"
            "Total: under 100 words. Be encouraging. Use friendly Indian English."
        ),
        user=(
            f"Question: {req.question}\n"
            f"Student's answer: {req.answer}\n"
            f"Relevant content: {ctx[:400]}"
        ),
    )
    out = llm(eval_prompt, max_tokens=230)["choices"][0]["text"].strip()
    return JSONResponse({"feedback": out})


# ── Reports ────────────────────────────────────────────────────────────────────

@app.get("/api/report")
def api_report():
    data = load_json(DATA_DIR / "final_report.json")
    if data is None:
        raise HTTPException(503, "No report yet — complete the quiz first")
    return JSONResponse(data)


@app.get("/api/quiz_results")
def api_quiz_results():
    return JSONResponse(load_json(DATA_DIR / "quiz_results.json") or {})


class QuizResultsReq(BaseModel):
    results: dict


@app.post("/api/save_quiz_results")
def api_save_quiz_results(req: QuizResultsReq):
    out = DATA_DIR / "quiz_results.json"
    DATA_DIR.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(req.results, f, indent=2, ensure_ascii=False)
    return JSONResponse({"status": "saved"})


class FinalReportReq(BaseModel):
    quiz_summary:  dict
    weak_topics:   list
    desc_evals:    list


@app.post("/api/generate_report")
def api_generate_report(req: FinalReportReq):
    topics_data = load_json(DATA_DIR / "processed_topics.json") or []
    topic_titles = [t["topic_title"] for t in topics_data]

    report_prompt = phi_prompt(
        system=(
            "You are an intelligent tutoring system generating a student learning report.\n"
            "Write a friendly, encouraging report (under 150 words) with clear sections:\n"
            "  - Topics Learned\n"
            "  - Quiz Performance\n"
            "  - Top 2 Strengths\n"
            "  - Top 2 Areas to Improve\n"
            "  - 2 Study Recommendations\n"
            "Use warm, motivating Indian English. Address the student directly."
        ),
        user=(
            f"Topics covered: {topic_titles}\n"
            f"Quiz summary: {json.dumps(req.quiz_summary)}\n"
            f"Weak areas (wrong questions): {req.weak_topics[:5]}"
        ),
    )
    report_text = llm(report_prompt, max_tokens=400)["choices"][0]["text"].strip()

    final = {
        "topics":                  topic_titles,
        "quiz_summary":            req.quiz_summary,
        "descriptive_evaluations": req.desc_evals,
        "weak_topics":             list(set(req.weak_topics)),
        "final_report":            report_text,
    }
    out = DATA_DIR / "final_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    return JSONResponse(final)


@app.get("/api/study_notes")
def api_study_notes():
    """Generate concise study notes for each topic. Cached in data/study_notes.json."""
    cache = DATA_DIR / "study_notes.json"
    if cache.exists():
        return JSONResponse(load_json(cache))

    topics_data = load_json(DATA_DIR / "processed_topics.json") or []
    notes = []
    for t in topics_data:
        title  = t.get("topic_title", "Topic")
        speech = (t.get("teacher_notes") or t.get("teacher_speech") or "")[:600]
        board  = t.get("board_notes", [])

        prompt = phi_prompt(
            system=(
                "You are a teacher writing concise, exam-ready study notes for a student.\n"
                "Format EXACTLY as:\n"
                "SUMMARY: 2 sentences summarising the topic.\n"
                "KEY POINTS:\n• point 1\n• point 2\n• point 3\n• point 4\n"
                "REMEMBER: One memorable tip or mnemonic.\n"
                "Keep total under 100 words. Clear, simple language."
            ),
            user=(
                f"Topic: {title}\n"
                f"Lesson content: {speech}\n"
                f"Board notes: {'; '.join(board[:5])}"
            ),
        )
        text = llm(prompt, max_tokens=220)["choices"][0]["text"].strip()
        notes.append({"topic_id": t["topic_id"], "title": title, "notes": text})

    with open(cache, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    return JSONResponse(notes)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    return JSONResponse({
        "status": "ok",
        "checks": {
            "model":            Path(MODEL_PATH).exists(),
            "processed_topics": (DATA_DIR / "processed_topics.json").exists(),
            "mcqs":             (DATA_DIR / "mcqs.json").exists(),
            "rag_store":        (DATA_DIR / "rag_store").exists(),
        },
    })