"""
STEP 4 — RAG Store Builder
══════════════════════════════════════════════════════════════════════
Builds a ChromaDB vector store from processed_topics.json so that
the API server can do retrieval-augmented generation (RAG) when
answering student doubts, giving feedback, etc.

Input:  data/processed_topics.json
Output: data/rag_store/  (ChromaDB persistent store)

Run this ONCE after step_3, before launching the server.
"""

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR    = Path("data")
INPUT_PATH  = DATA_DIR / "processed_topics.json"
RAG_DIR     = DATA_DIR / "rag_store"

def main():
    print("\n=== STEP 4: Building RAG Store ===\n")

    if not INPUT_PATH.exists():
        print(f"❌ {INPUT_PATH} not found — run step_3 first.")
        return

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        topics = json.load(f)

    if not topics:
        print("⚠️  No topics found — nothing to index.")
        return

    print(f"📂 Loaded {len(topics)} topics from {INPUT_PATH}")

    # ── Init ChromaDB ────────────────────────────────────────────────
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    client  = chromadb.PersistentClient(path=str(RAG_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # Delete existing collection to rebuild fresh
    try:
        client.delete_collection("topics")
        print("🗑️  Deleted old 'topics' collection (rebuilding fresh)")
    except Exception:
        pass

    collection = client.create_collection(
        name="topics",
        embedding_function=embed_fn
    )

    # ── Index each topic in chunks ───────────────────────────────────
    ids, docs, metas = [], [], []

    for t in topics:
        tid   = str(t["topic_id"])
        title = t.get("topic_title", f"Topic {tid}")
        notes = t.get("teacher_notes") or t.get("teacher_speech") or ""
        board = t.get("board_notes", [])
        board_str = " | ".join(board) if board else ""

        # Chunk 1: full teacher notes (primary content)
        if notes:
            ids.append(f"topic_{tid}_notes")
            docs.append(notes[:1000])
            metas.append({"topic_id": int(tid), "title": title, "chunk": "notes"})

        # Chunk 2: board summary points
        if board_str:
            ids.append(f"topic_{tid}_board")
            docs.append(f"{title}: {board_str}")
            metas.append({"topic_id": int(tid), "title": title, "chunk": "board"})

        # Chunk 3: raw transcript texts
        raw_texts = " ".join(t.get("texts", []))
        if raw_texts:
            ids.append(f"topic_{tid}_raw")
            docs.append(raw_texts[:800])
            metas.append({"topic_id": int(tid), "title": title, "chunk": "raw"})

    if not ids:
        print("⚠️  No content to index.")
        return

    # Add in batches of 50 (ChromaDB default limit)
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            documents=docs[i:i+BATCH],
            metadatas=metas[i:i+BATCH]
        )
        print(f"   Indexed chunks {i+1}–{min(i+BATCH, len(ids))} of {len(ids)}")

    print(f"\n✅ RAG store built — {len(ids)} chunks indexed → {RAG_DIR}")
    print("   Run step_5_mcq_generation.py next.\n")


if __name__ == "__main__":
    main()