import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# CONFIG
# =========================

DATA_DIR = Path("data")
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
OUTPUT_PATH = DATA_DIR / "topics.json"

SIMILARITY_THRESHOLD = 0.65
MIN_TOPIC_DURATION = 45
MAX_TOPIC_DURATION = 180
MIN_SENTENCE_LENGTH = 20
ROLLING_WINDOW = 3
LOW_SIMILARITY_PATIENCE = 2

MERGE_MIN_DURATION = 30
MERGE_MIN_COHERENCE = 0.4

# =========================
# LOAD TRANSCRIPT
# =========================

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)

texts = [seg["text"] for seg in transcript]

# =========================
# EMBEDDINGS
# =========================

print("🔎 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# =========================
# HELPERS
# =========================

def rolling_embedding(embeds, window=ROLLING_WINDOW):
    return np.mean(embeds[-window:], axis=0)

def topic_duration(topic):
    return topic["end"] - topic["start"]

# =========================
# SEGMENTATION
# =========================

topics = []
current = {
    "start": transcript[0]["start"],
    "end": transcript[0]["end"],
    "texts": [texts[0]],
    "embeddings": [embeddings[0]],
    "similarities": []
}

low_similarity_count = 0

for i in range(1, len(transcript)):
    text = texts[i]

    if len(text) < MIN_SENTENCE_LENGTH:
        current["texts"].append(text)
        current["end"] = transcript[i]["end"]
        current["embeddings"].append(embeddings[i])
        continue

    avg_embed = rolling_embedding(current["embeddings"])
    similarity = cosine_similarity([avg_embed], [embeddings[i]])[0][0]
    current["similarities"].append(similarity)

    duration = topic_duration(current)
    dynamic_threshold = max(
        0.55,
        np.mean(current["similarities"][-3:]) - 0.05
    )

    if similarity < dynamic_threshold:
        low_similarity_count += 1
    else:
        low_similarity_count = 0

    should_split = (
        (low_similarity_count >= LOW_SIMILARITY_PATIENCE and duration >= MIN_TOPIC_DURATION)
        or duration >= MAX_TOPIC_DURATION
    )

    if should_split:
        topics.append({
            "start": current["start"],
            "end": current["end"],
            "texts": current["texts"],
            "coherence_score": round(
                float(np.mean(current["similarities"])) if current["similarities"] else 1.0,
                3
            )
        })

        current = {
            "start": transcript[i]["start"],
            "end": transcript[i]["end"],
            "texts": [text],
            "embeddings": [embeddings[i]],
            "similarities": []
        }
        low_similarity_count = 0
    else:
        current["end"] = transcript[i]["end"]
        current["texts"].append(text)
        current["embeddings"].append(embeddings[i])

# =========================
# FINAL TOPIC
# =========================

topics.append({
    "start": current["start"],
    "end": current["end"],
    "texts": current["texts"],
    "coherence_score": round(
        float(np.mean(current["similarities"])) if current["similarities"] else 1.0,
        3
    )
})

# =========================
# MERGE SMALL TOPICS
# =========================

def merge_small_topics(topics):
    merged = [topics[0]]

    for topic in topics[1:]:
        last = merged[-1]
        duration = topic["end"] - topic["start"]

        if duration < MERGE_MIN_DURATION or topic["coherence_score"] < MERGE_MIN_COHERENCE:
            last["end"] = topic["end"]
            last["texts"].extend(topic["texts"])
            last["coherence_score"] = round(
                (last["coherence_score"] + topic["coherence_score"]) / 2, 3
            )
        else:
            merged.append(topic)

    return merged

topics = merge_small_topics(topics)

# =========================
# SAVE
# =========================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(topics, f, indent=2, ensure_ascii=False)

print("✅ STEP 2 complete!")
print(f"📚 Generated {len(topics)} topics")
