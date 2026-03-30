"""
STEP 2 — Concept-Based Semantic Topic Segmentation
══════════════════════════════════════════════════════════════════════
Pipeline overview
─────────────────
1. CLEAN      Strip non-educational filler (greetings, CTAs, sign-offs).
2. EMBED      Encode every cleaned segment with a sentence-transformer.
3. SEGMENT    Detect concept boundaries using three complementary signals:
                a) Cosine-similarity drop between adjacent embeddings
                b) Concept-class transition (definition → process → …)
                c) Strong boundary keyword at sentence start
4. POST-PROCESS
   • Merge topics with fewer than MIN_TOPIC_WORDS (30) into their neighbour
   • Merge topics that are semantically very close (sim > MERGE_SIM_FLOOR)
   • Cap total topic count for short videos (≤ MAX_TOPICS_SHORT_VIDEO)
   • Re-split any topic that grew beyond MAX_TOPIC_WORDS (150)
5. SCORE & OUTPUT
   Each output topic carries: start, end, texts, word_count, topic_type,
   quality_score, is_meaningful.

Design constraints (from spec)
───────────────────────────────
• Target word range per topic : 40 – 150 words
• Merge threshold             : < 30 words  → merge with neighbour
• Max topics for short video  : 6–8  (video duration ≤ SHORT_VIDEO_SECS)
• Prefer fewer, coherent topics over many tiny fragments
• Never split on sentence count or time duration alone
"""

import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR        = Path("data")
TRANSCRIPT_PATH = DATA_DIR / "transcript.json"
OUTPUT_PATH     = DATA_DIR / "topics.json"

# ══════════════════════════════════════════════════════════════════════
# TUNABLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# ── Segmentation thresholds ───────────────────────────────────────────
# Primary split: adjacent cosine similarity drops below this → new topic
SIM_SPLIT_THRESHOLD   = 0.50

# Concept-class split only fires when similarity is also below this
CLASS_SPLIT_SIM_CAP   = 0.68

# Minimum segments in a group before any split is allowed
MIN_SEGS_BEFORE_SPLIT = 2

# ── Post-processing word limits ───────────────────────────────────────
MIN_TOPIC_WORDS       = 30    # merge if below
TARGET_MIN_WORDS      = 40    # soft lower bound (for quality scoring)
TARGET_MAX_WORDS      = 150   # re-split if above
MERGE_SMALL_WORDS     = 30    # alias for clarity

# ── Topic-count cap ───────────────────────────────────────────────────
SHORT_VIDEO_SECS      = 180   # ≤ 3 minutes → apply topic cap
MAX_TOPICS_SHORT_VIDEO = 8
MAX_TOPICS_LONG_VIDEO  = 14   # loose cap for longer content

# ── Post-merge similarity: merge two adjacent topics if their centroid
#    similarity is above this (they're still the same concept)
MERGE_SIM_FLOOR       = 0.72

# ── Re-split: topic > TARGET_MAX_WORDS uses a lower threshold
RESPLIT_SIM_THRESHOLD = 0.58

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — TRANSCRIPT CLEANING
# ══════════════════════════════════════════════════════════════════════

NON_EDUCATIONAL_PATTERNS = [
    # Social / channel CTAs
    r'\b(subscribe|subscriber|subscribing)\b',
    r'\b(click\s+(?:the\s+)?bell|bell\s+icon|hit\s+the\s+bell|notification)\b',
    r'\b(like\s+(?:this\s+)?video|hit\s+like|press\s+like|smash\s+like)\b',
    r'\b(share\s+(?:this\s+)?video|share\s+with|share\s+it)\b',
    r'\b(comment\s+below|leave\s+a\s+comment|drop\s+a\s+comment)\b',
    r'\b(thanks?\s+for\s+watching|thank\s+you\s+for\s+watching|thanks?\s+for\s+tuning)\b',
    r'\b(see\s+you\s+(?:in\s+the\s+)?next|next\s+video|next\s+episode|next\s+lesson)\b',
    r'\b(bye+|goodbye|take\s+care|stay\s+tuned|that\'s\s+all\s+for\s+today)\b',
    # Greetings / openers
    r'^\s*(hello+|hi+|hey+)[,!\s]',
    r'^\s*welcome\s+(back\s+)?to\b',
    r'^\s*good\s+(morning|afternoon|evening)\b',
    r'^\s*today\s+we\s+(?:are\s+going\s+to|will|\'ll)\s+(?:learn|study|cover|discuss|talk)\b',
    # Sponsors / promos
    r'\b(patreon|sponsor|ad\b|advertisement|promotion|coupon|discount\s+code)\b',
    r'\b(check\s+out|visit\s+our|follow\s+us\s+on|link\s+in\s+(?:the\s+)?(?:bio|description))\b',
    r'\b(instagram|twitter|facebook|youtube\s+channel|social\s+media)\b',
]

EDUCATIONAL_SIGNALS = [
    r'\b(is\s+(?:a|an|the)|are\s+(?:a|an|the)|refers?\s+to|defined?\s+as|means?\s+that)\b',
    r'\b(process|cycle|stage|step|phase|method|technique|principle|law|theory|mechanism)\b',
    r'\b(cause|effect|result|leads?\s+to|because|therefore|hence|thus|consequently)\b',
    r'\b(example|instance|for\s+(?:example|instance)|such\s+as|e\.g\.|namely|including)\b',
    r'\b(type|kind|category|class|group|form|variety|classification)\b',
    r'\b(compare|contrast|difference|similar|unlike|whereas|while\b|versus|vs\.)\b',
    r'\b(important|key|main|primary|major|essential|critical|significant|notable)\b',
    r'\b(first|second|third|fourth|finally|next|then|after|before|during|when|once)\b',
    r'\b(increase|decrease|change|grow|reduce|form|create|produce|release|absorb|convert)\b',
    r'\b(property|properties|characteristic|feature|attribute|consist\s+of|composed\s+of)\b',
    r'\b(water|earth|sun|heat|temperature|pressure|energy|force|matter|molecule|atom)\b',
    r'\b(define|explain|describe|illustrate|demonstrate|show|prove|represent)\b',
]


def _is_non_educational(text: str) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in NON_EDUCATIONAL_PATTERNS)


def _has_educational_value(text: str) -> bool:
    t = text.lower()
    if any(re.search(p, t) for p in EDUCATIONAL_SIGNALS):
        return True
    return len(text.split()) >= 7


def _clean_text(text: str) -> str:
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_transcript(segments: list) -> list:
    """
    Walk every raw segment and keep only those that carry genuine
    educational content.  Returns cleaned segment dicts.
    """
    kept = []
    for seg in segments:
        raw = seg.get("text", "").strip()
        if not raw:
            continue
        if _is_non_educational(raw):
            continue
        cleaned = _clean_text(raw)
        if not _has_educational_value(cleaned):
            continue
        if len(cleaned.split()) < 4:
            continue
        copy = seg.copy()
        copy["text"] = cleaned
        kept.append(copy)
    return kept


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — CONCEPT-CLASS DETECTION
# ══════════════════════════════════════════════════════════════════════

_CONCEPT_RULES: list[tuple[str, list[str]]] = [
    ("definition",   [
        r'\b(is\s+(?:a|an|the)\b|defined?\s+as|refers?\s+to|'
        r'means?\s+that|what\s+is\b|meaning\s+of|concept\s+of|'
        r'term\s+(?:refers|means)|called\s+(?:a|an|the))\b',
    ]),
    ("process",      [
        r'\b(process|cycle|stage|step|phase|sequence|procedure|'
        r'mechanism|how\s+(?:it\s+)?(?:works?|happens?)|'
        r'(?:first|second|third|finally)\b.*\b(?:then|next)\b|'
        r'steps?\s+(?:are|in|of|involved))\b',
    ]),
    ("cause_effect", [
        r'\b(cause[sd]?|because|therefore|hence|thus|consequently|'
        r'leads?\s+to|results?\s+in|due\s+to|effect\s+of|'
        r'reason\s+(?:is|for)|responsible\s+for)\b',
    ]),
    ("example",      [
        r'\b(for\s+(?:example|instance)|such\s+as|e\.g\.|namely|'
        r'consider\s+(?:this|the\s+case)|like\b|imagine\b|'
        r'suppose\b|take\s+the\s+case)\b',
    ]),
    ("comparison",   [
        r'\b(compare[sd]?|contrast[sd]?|difference|similar(?:ly)?|'
        r'unlike|whereas|on\s+the\s+other\s+hand|'
        r'while\b|both\b|versus|vs\.)\b',
    ]),
    ("properties",   [
        r'\b(propert(?:y|ies)|characteristic[s]?|feature[s]?|'
        r'attribute[s]?|consist\s+of|made\s+of|composed\s+of|'
        r'contain[s]?|made\s+up\s+of|structure\s+of)\b',
    ]),
]


def detect_concept_class(texts: list[str]) -> str:
    """Return the dominant pedagogical class of a group of sentences."""
    combined = " ".join(texts).lower()
    for label, patterns in _CONCEPT_RULES:
        for pat in patterns:
            if re.search(pat, combined):
                return label
    return "general"


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — BOUNDARY KEYWORD DETECTOR
# ══════════════════════════════════════════════════════════════════════

_BOUNDARY_OPENERS = re.compile(
    r'^\s*(?:'
    r'now\s+(?:let\s+us|we\s+will|let\'s|i\s+will)\b|'
    r'moving\s+(?:on|ahead|forward)\b|'
    r'(?:next|another|the\s+(?:second|third|fourth|next))\s+'
    r'(?:topic|concept|idea|point|thing|stage|step|part)\b|'
    r'let\s+(?:us|me)\s+(?:now\s+)?(?:talk|discuss|look|understand|explain|consider)\b|'
    r'(?:apart\s+from|besides|in\s+addition\s+to|furthermore|moreover)\b|'
    r'on\s+the\s+other\s+hand\b|'
    r'in\s+contrast(?:\s+to)?\b|'
    r'to\s+(?:summarize|summarise|conclude|recap)\b|'
    r'in\s+(?:conclusion|summary|short)\b|'
    r'so\s+(?:now|let\'s|we)\b'
    r')',
    re.IGNORECASE,
)


def _starts_new_concept(text: str) -> bool:
    return bool(_BOUNDARY_OPENERS.match(text))


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — INITIAL SEGMENTATION (semantic boundaries only)
# ══════════════════════════════════════════════════════════════════════

def _centroid(indices: list[int], embeddings: np.ndarray) -> np.ndarray:
    """Mean embedding of a set of segment indices."""
    return embeddings[indices].mean(axis=0)


def _topic_word_count(txts: list[str]) -> int:
    return len(" ".join(txts).split())


def _initial_segment(
    filtered_segments: list,
    embeddings: np.ndarray,
) -> list[dict]:
    """
    First pass: split only on genuine semantic boundaries.
    Returns a list of raw topic dicts (no scoring yet).
    """
    if not filtered_segments:
        return []

    topics   : list[dict] = []
    cur_segs : list       = [filtered_segments[0]]
    cur_txts : list[str]  = [filtered_segments[0]["text"]]
    cur_idxs : list[int]  = [0]

    for i in range(1, len(filtered_segments)):
        seg  = filtered_segments[i]
        text = seg["text"]

        # ── Signal 1: pairwise cosine similarity ──────────────────────
        sim_adj = float(
            cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0]
        )

        # ── Signal 2: concept-class transition ────────────────────────
        prev_cls     = detect_concept_class(cur_txts)
        next_cls     = detect_concept_class([text])
        class_change = (prev_cls != next_cls and next_cls != "general")

        # ── Signal 3: boundary keyword ────────────────────────────────
        kw_bound = _starts_new_concept(text)

        # ── Decision ──────────────────────────────────────────────────
        split = False
        if len(cur_segs) >= MIN_SEGS_BEFORE_SPLIT:
            if sim_adj < SIM_SPLIT_THRESHOLD:
                split = True
            elif class_change and sim_adj < CLASS_SPLIT_SIM_CAP:
                split = True
            elif kw_bound:
                split = True

        if split:
            topics.append(_build_raw_topic(cur_segs, cur_txts, cur_idxs))
            cur_segs = [seg]
            cur_txts = [text]
            cur_idxs = [i]
        else:
            cur_segs.append(seg)
            cur_txts.append(text)
            cur_idxs.append(i)

    if cur_segs:
        topics.append(_build_raw_topic(cur_segs, cur_txts, cur_idxs))

    return topics


def _build_raw_topic(
    segs: list, txts: list[str], idxs: list[int]
) -> dict:
    return {
        "start":         segs[0].get("start", 0),
        "end":           segs[-1].get("end",   0),
        "texts":         list(txts),
        "words":         [w for s in segs for w in s.get("words", [])],
        "topic_type":    detect_concept_class(txts),
        "segment_count": len(segs),
        "_seg_indices":  list(idxs),   # used during post-processing, removed at end
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════

# ── 5a. Merge topics that are too small ──────────────────────────────

def _merge_small_topics(
    topics: list[dict],
    embeddings: np.ndarray,
) -> list[dict]:
    """
    Repeatedly scan for topics below MIN_TOPIC_WORDS.
    Merge each one with whichever neighbour is semantically closer.
    Continue until no topic is below the threshold (or only 1 remains).
    """
    changed = True
    while changed and len(topics) > 1:
        changed = False
        merged  = []
        i = 0
        while i < len(topics):
            t = topics[i]
            wc = _topic_word_count(t["texts"])

            if wc < MERGE_SMALL_WORDS and len(topics) > 1:
                # Find the closer neighbour by centroid similarity
                prev_t  = merged[-1]  if merged            else None
                next_t  = topics[i+1] if i + 1 < len(topics) else None

                if prev_t is None:
                    # Only a next neighbour → merge forward
                    combined = _merge_two(t, next_t)
                    merged.append(combined)
                    i += 2
                elif next_t is None:
                    # Only a prev neighbour → merge backward
                    merged[-1] = _merge_two(prev_t, t)
                    i += 1
                else:
                    # Both exist: pick the one with higher centroid sim
                    c_prev = _centroid_sim_between(prev_t, t, embeddings)
                    c_next = _centroid_sim_between(t, next_t, embeddings)
                    if c_prev >= c_next:
                        merged[-1] = _merge_two(prev_t, t)
                    else:
                        combined = _merge_two(t, next_t)
                        merged.append(combined)
                        i += 2
                        continue
                changed = True
            else:
                merged.append(t)
            i += 1
        topics = merged

    return topics


def _centroid_sim_between(t1: dict, t2: dict, embeddings: np.ndarray) -> float:
    """Cosine similarity between the centroid embeddings of two topics."""
    idx1 = t1.get("_seg_indices", [])
    idx2 = t2.get("_seg_indices", [])
    if not idx1 or not idx2:
        return 0.0
    c1 = embeddings[idx1].mean(axis=0)
    c2 = embeddings[idx2].mean(axis=0)
    return float(cosine_similarity([c1], [c2])[0][0])


def _merge_two(t1: dict, t2: dict) -> dict:
    """Combine two adjacent topic dicts into one."""
    merged_txts = t1["texts"] + t2["texts"]
    merged_idxs = t1.get("_seg_indices", []) + t2.get("_seg_indices", [])
    return {
        "start":         t1["start"],
        "end":           t2["end"],
        "texts":         merged_txts,
        "words":         t1["words"] + t2["words"],
        "topic_type":    detect_concept_class(merged_txts),
        "segment_count": t1["segment_count"] + t2["segment_count"],
        "_seg_indices":  merged_idxs,
    }


# ── 5b. Merge adjacent topics that are semantically very similar ──────

def _merge_similar_topics(
    topics: list[dict],
    embeddings: np.ndarray,
) -> list[dict]:
    """
    If two adjacent topics have centroid similarity > MERGE_SIM_FLOOR
    they likely belong to the same concept → merge them.
    Only fires when both topics are also within target word range
    after merging.
    """
    changed = True
    while changed and len(topics) > 1:
        changed = False
        merged  = []
        i = 0
        while i < len(topics):
            if i + 1 < len(topics):
                sim = _centroid_sim_between(topics[i], topics[i + 1], embeddings)
                combined_wc = (
                    _topic_word_count(topics[i]["texts"])
                    + _topic_word_count(topics[i + 1]["texts"])
                )
                if sim > MERGE_SIM_FLOOR and combined_wc <= TARGET_MAX_WORDS:
                    merged.append(_merge_two(topics[i], topics[i + 1]))
                    i += 2
                    changed = True
                    continue
            merged.append(topics[i])
            i += 1
        topics = merged
    return topics


# ── 5c. Cap total topic count for short videos ───────────────────────

def _cap_topic_count(
    topics: list[dict],
    embeddings: np.ndarray,
    video_duration_secs: float,
) -> list[dict]:
    """
    For short videos (≤ SHORT_VIDEO_SECS), repeatedly merge the pair of
    adjacent topics with the highest centroid similarity until the count
    is within the cap.
    """
    cap = (
        MAX_TOPICS_SHORT_VIDEO
        if video_duration_secs <= SHORT_VIDEO_SECS
        else MAX_TOPICS_LONG_VIDEO
    )

    while len(topics) > cap:
        # Find the pair with the highest centroid similarity
        best_sim = -1.0
        best_i   = 0
        for i in range(len(topics) - 1):
            sim = _centroid_sim_between(topics[i], topics[i + 1], embeddings)
            if sim > best_sim:
                best_sim = sim
                best_i   = i

        # Merge that pair
        merged = []
        for j, t in enumerate(topics):
            if j == best_i:
                merged.append(_merge_two(t, topics[j + 1]))
            elif j == best_i + 1:
                pass   # already consumed
            else:
                merged.append(t)
        topics = merged

    return topics


# ── 5d. Re-split topics that grew too large ───────────────────────────

def _resplit_large_topics(
    topics: list[dict],
    filtered_segments: list,
    embeddings: np.ndarray,
) -> list[dict]:
    """
    For any topic exceeding TARGET_MAX_WORDS, attempt to find the
    best internal split point (largest cosine drop) and split there.
    Repeat until no topic is too large.
    """
    output = []
    for topic in topics:
        wc = _topic_word_count(topic["texts"])
        if wc <= TARGET_MAX_WORDS:
            output.append(topic)
            continue

        # Try to find the best split point inside this topic's segments
        idxs = topic.get("_seg_indices", [])
        if len(idxs) < 4:
            # Too few segments to re-split meaningfully
            output.append(topic)
            continue

        best_drop = 0.0
        best_cut  = -1
        for k in range(1, len(idxs) - 1):
            sim = float(
                cosine_similarity(
                    [embeddings[idxs[k]]],
                    [embeddings[idxs[k - 1]]]
                )[0][0]
            )
            drop = 1.0 - sim
            # Only split if the resulting halves are ≥ MIN_TOPIC_WORDS
            left_wc  = _topic_word_count(topic["texts"][:k])
            right_wc = _topic_word_count(topic["texts"][k:])
            if drop > best_drop and left_wc >= MIN_TOPIC_WORDS and right_wc >= MIN_TOPIC_WORDS:
                best_drop = drop
                best_cut  = k

        if best_cut == -1 or best_drop < (1.0 - RESPLIT_SIM_THRESHOLD):
            # No good split point found
            output.append(topic)
            continue

        # Build two sub-topics from the cut
        left_segs  = [filtered_segments[j] for j in idxs[:best_cut]]
        right_segs = [filtered_segments[j] for j in idxs[best_cut:]]
        left_txts  = topic["texts"][:best_cut]
        right_txts = topic["texts"][best_cut:]

        left_topic  = _build_raw_topic(left_segs,  left_txts,  idxs[:best_cut])
        right_topic = _build_raw_topic(right_segs, right_txts, idxs[best_cut:])

        # Recursively check if either half still needs splitting
        for sub in _resplit_large_topics([left_topic, right_topic], filtered_segments, embeddings):
            output.append(sub)

    return output


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — QUALITY SCORING & FINAL FIELD ASSEMBLY
# ══════════════════════════════════════════════════════════════════════

def _score_and_finalise(topics: list[dict]) -> list[dict]:
    """
    Attach quality_score, is_meaningful, word_count.
    Remove internal helper fields (_seg_indices).
    """
    final = []
    for idx, t in enumerate(topics):
        combined   = " ".join(t["texts"])
        word_count = len(combined.split())

        # Concept density: how many educational-signal patterns fire
        signal_hits = sum(
            1 for p in EDUCATIONAL_SIGNALS
            if re.search(p, combined, re.IGNORECASE)
        )
        density      = min(1.0, signal_hits / max(len(EDUCATIONAL_SIGNALS), 1))
        length_score = min(1.0, word_count / TARGET_MIN_WORDS)
        quality      = round(0.60 * length_score + 0.40 * density, 3)

        entry = {
            "topic_id":      idx,
            "start":         t["start"],
            "end":           t["end"],
            "texts":         t["texts"],
            "word_count":    word_count,
            "topic_type":    t["topic_type"],
            "segment_count": t["segment_count"],
            "words":         t.get("words", []),
            "quality_score": quality,
            "is_meaningful": quality > 0.30 and word_count >= TARGET_MIN_WORDS,
        }
        final.append(entry)
    return final


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def segment_transcript(
    filtered_segments: list,
    embeddings: np.ndarray,
    video_duration_secs: float = 0.0,
) -> list[dict]:
    """
    Full segmentation pipeline.

    Parameters
    ----------
    filtered_segments   : cleaned segment dicts (output of filter_transcript)
    embeddings          : sentence-transformer embeddings, one per segment
    video_duration_secs : total video duration (used for topic-count cap)

    Returns
    -------
    List of topic dicts ready for downstream processing.
    """
    if not filtered_segments:
        return []

    # ── 1. Initial semantic segmentation ─────────────────────────────
    topics = _initial_segment(filtered_segments, embeddings)
    print(f"   Initial segments: {len(topics)}")

    # ── 2. Merge topics below minimum word count ──────────────────────
    topics = _merge_small_topics(topics, embeddings)
    print(f"   After merging small topics: {len(topics)}")

    # ── 3. Merge semantically similar adjacent topics ─────────────────
    topics = _merge_similar_topics(topics, embeddings)
    print(f"   After merging similar topics: {len(topics)}")

    # ── 4. Cap topic count for short videos ───────────────────────────
    if video_duration_secs > 0:
        topics = _cap_topic_count(topics, embeddings, video_duration_secs)
        print(f"   After applying topic-count cap: {len(topics)}")

    # ── 5. Re-split any topic that grew too large ─────────────────────
    topics = _resplit_large_topics(topics, filtered_segments, embeddings)
    print(f"   After re-splitting large topics: {len(topics)}")

    # ── 6. Score and assemble final output ────────────────────────────
    return _score_and_finalise(topics)


# ══════════════════════════════════════════════════════════════════════
# MAIN — standalone execution
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load ──────────────────────────────────────────────────────────
    print("📂  Loading transcript…")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    print(f"    Raw segments : {len(transcript)}")

    # ── Infer video duration from last segment end time ───────────────
    video_duration = 0.0
    for seg in reversed(transcript):
        if "end" in seg:
            video_duration = float(seg["end"])
            break
    print(f"    Video length : {video_duration:.1f}s  "
          f"({'short' if video_duration <= SHORT_VIDEO_SECS else 'long'})")

    # ── Clean ─────────────────────────────────────────────────────────
    print("\n🔍  Filtering non-educational content…")
    filtered = filter_transcript(transcript)
    print(f"    Educational segments kept : {len(filtered)}")

    if not filtered:
        print("❌  No educational content — writing empty output.")
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        raise SystemExit(0)

    # ── Embed ─────────────────────────────────────────────────────────
    print("\n🧠  Encoding segments with sentence-transformer…")
    model      = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        [s["text"] for s in filtered],
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"    Encoded {len(embeddings)} segments.")

    # ── Segment ───────────────────────────────────────────────────────
    print("\n✂️   Running segmentation pipeline…")
    topics = segment_transcript(filtered, embeddings, video_duration)

    # ── Print summary ─────────────────────────────────────────────────
    meaningful = sum(1 for t in topics if t["is_meaningful"])
    print(f"\n✅  {len(topics)} topics  ({meaningful} meaningful)")
    print("─" * 72)
    print(
        f"  {'#':>2}  {'type':<13}  {'segs':>4}  {'words':>5}  "
        f"{'quality':>7}  {'start':>7}  {'end':>7}  ok?"
    )
    print("─" * 72)
    for t in topics:
        dur = f"{t['start']:.1f}s"
        end = f"{t['end']:.1f}s"
        ok  = "✓" if t["is_meaningful"] else "·"
        print(
            f"  {t['topic_id']+1:>2}  [{t['topic_type']:<11}]  "
            f"{t['segment_count']:>4}  {t['word_count']:>5}  "
            f"{t['quality_score']:>7.3f}  {dur:>7}  {end:>7}  {ok}"
        )
    print("─" * 72)

    # Word-count band check
    under = [t for t in topics if t["word_count"] < TARGET_MIN_WORDS]
    over  = [t for t in topics if t["word_count"] > TARGET_MAX_WORDS]
    if under:
        print(f"  ⚠  {len(under)} topic(s) below {TARGET_MIN_WORDS} words "
              f"(may need manual review)")
    if over:
        print(f"  ⚠  {len(over)} topic(s) above {TARGET_MAX_WORDS} words "
              f"(may need manual review)")

    # ── Save ──────────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(topics, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Saved → {OUTPUT_PATH}")