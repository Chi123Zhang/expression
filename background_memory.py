import os
import json
import sqlite3
import pickle
from typing import Dict, List, Any

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


DB_PATH = "background_memory.db"
VECTOR_INDEX_PATH = "background_faiss.index"
VECTOR_META_PATH = "background_faiss_meta.pkl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# OpenAI / Embedding helpers
# -----------------------------
def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _get_embedder() -> SentenceTransformer:
    if not hasattr(_get_embedder, "_model"):
        _get_embedder._model = SentenceTransformer(EMBED_MODEL_NAME)
    return _get_embedder._model


def _parse_json_safely(text: str) -> Dict:
    if not text:
        raise ValueError("Empty response from model.")

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


def _embed_texts(texts: List[str]) -> np.ndarray:
    model = _get_embedder()
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")


# -----------------------------
# DB init
# -----------------------------
def _init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            structured_profile_json TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS background_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            chunk_id TEXT,
            chunk_type TEXT,
            chunk_text TEXT,
            source_type TEXT,
            retrieval_priority REAL
        )
    """)

    conn.commit()
    conn.close()


# -----------------------------
# Vector store helpers
# -----------------------------
def _load_vector_store():
    if os.path.exists(VECTOR_INDEX_PATH) and os.path.exists(VECTOR_META_PATH):
        index = faiss.read_index(VECTOR_INDEX_PATH)
        with open(VECTOR_META_PATH, "rb") as f:
            meta = pickle.load(f)
        return index, meta
    return None, []


def _save_vector_store(index, meta) -> None:
    faiss.write_index(index, VECTOR_INDEX_PATH)
    with open(VECTOR_META_PATH, "wb") as f:
        pickle.dump(meta, f)


def _rebuild_user_vectors(user_id: str) -> None:
    """
    Rebuild the FAISS index from DB rows.
    Simpler and safer for MVP / small-scale multi-user demo.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT user_id, chunk_id, chunk_type, chunk_text, source_type, retrieval_priority
        FROM background_chunks
    """)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        if os.path.exists(VECTOR_INDEX_PATH):
            os.remove(VECTOR_INDEX_PATH)
        if os.path.exists(VECTOR_META_PATH):
            os.remove(VECTOR_META_PATH)
        return

    texts = [r[3] for r in rows]
    vecs = _embed_texts(texts)
    dim = vecs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    meta = []
    for r in rows:
        meta.append({
            "user_id": r[0],
            "chunk_id": r[1],
            "chunk_type": r[2],
            "chunk_text": r[3],
            "source_type": r[4],
            "retrieval_priority": r[5],
        })

    _save_vector_store(index, meta)


# -----------------------------
# Raw input combination
# -----------------------------
def _combine_raw_background(raw_background_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    sources = []
    all_texts = []

    for item in raw_background_inputs:
        source_type = item.get("source_type", "unknown")
        raw_text = item.get("raw_text", "").strip()
        if raw_text:
            sources.append({
                "source_type": source_type,
                "raw_text": raw_text
            })
            all_texts.append(raw_text)

    return {
        "sources": sources,
        "combined_text": "\n\n".join(all_texts)
    }


# -----------------------------
# Parsing / normalization
# -----------------------------
def _fallback_profile(raw_text: str) -> Dict:
    text_lower = raw_text.lower()

    role_lens = "general"
    if any(x in text_lower for x in ["product manager", "pm", "product lead"]):
        role_lens = "pm"
    elif any(x in text_lower for x in ["engineer", "developer", "software", "machine learning", "data scientist"]):
        role_lens = "engineer"
    elif any(x in text_lower for x in ["business", "marketing", "strategy", "founder", "director"]):
        role_lens = "business"

    technical_depth = "medium"
    if any(x in text_lower for x in ["backend", "distributed systems", "machine learning", "python", "software engineer", "llm", "ai"]):
        technical_depth = "high"
    elif any(x in text_lower for x in ["beginner", "non-technical", "not deeply technical", "limited technical"]):
        technical_depth = "low"

    business_depth = "medium"
    if any(x in text_lower for x in ["strategy", "stakeholder", "marketing", "leadership", "business", "director", "founder"]):
        business_depth = "high"

    preferred_explanation_style = ["high_level"]
    if "step-by-step" in text_lower or "step by step" in text_lower:
        preferred_explanation_style = ["step_by_step"]
    if "concise" in text_lower:
        preferred_explanation_style.append("concise")
    if "analogy" in text_lower:
        preferred_explanation_style.append("analogy_driven")

    jargon_tolerance = "medium"
    if "minimal jargon" in text_lower or "avoid jargon" in text_lower:
        jargon_tolerance = "low"

    return {
        "current_role": role_lens,
        "role_lens": role_lens,
        "industry_domain": [],
        "technical_depth": technical_depth,
        "business_depth": business_depth,
        "preferred_explanation_style": preferred_explanation_style,
        "jargon_tolerance": jargon_tolerance,
        "strength_areas": [],
        "weak_areas": [],
        "current_projects": [],
        "short_reason": "Fallback profile generated from uploaded background text."
    }


def _parse_background_with_llm(raw_text: str) -> Dict:
    client = _get_openai_client()

    prompt = f"""
You are extracting a structured user profile from resume/background text.

Return ONLY valid JSON with exactly these keys:
- current_role
- role_lens
- industry_domain
- technical_depth
- business_depth
- preferred_explanation_style
- jargon_tolerance
- strength_areas
- weak_areas
- current_projects
- short_reason

Rules:
- role_lens must be one of: ["general", "pm", "engineer", "business"]
- technical_depth must be one of: ["low", "medium", "high"]
- business_depth must be one of: ["low", "medium", "high"]
- jargon_tolerance must be one of: ["low", "medium", "high"]
- industry_domain must be a list of strings
- preferred_explanation_style must be a list of strings
- strength_areas must be a list of strings
- weak_areas must be a list of strings
- current_projects must be a list of strings
- short_reason should be one sentence

Background text:
{raw_text[:7000]}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    return _parse_json_safely(content)


def _normalize_profile(profile: Dict) -> Dict:
    valid_roles = {"general", "pm", "engineer", "business"}
    valid_levels = {"low", "medium", "high"}

    def ensure_list(x):
        return x if isinstance(x, list) else []

    role_lens = profile.get("role_lens", "general")
    if role_lens not in valid_roles:
        role_lens = "general"

    technical_depth = profile.get("technical_depth", "medium")
    if technical_depth not in valid_levels:
        technical_depth = "medium"

    business_depth = profile.get("business_depth", "medium")
    if business_depth not in valid_levels:
        business_depth = "medium"

    jargon_tolerance = profile.get("jargon_tolerance", "medium")
    if jargon_tolerance not in valid_levels:
        jargon_tolerance = "medium"

    normalized = {
        "current_role": profile.get("current_role", role_lens),
        "role_lens": role_lens,
        "industry_domain": ensure_list(profile.get("industry_domain")),
        "technical_depth": technical_depth,
        "business_depth": business_depth,
        "preferred_explanation_style": ensure_list(profile.get("preferred_explanation_style")),
        "jargon_tolerance": jargon_tolerance,
        "strength_areas": ensure_list(profile.get("strength_areas")),
        "weak_areas": ensure_list(profile.get("weak_areas")),
        "current_projects": ensure_list(profile.get("current_projects")),
        "short_reason": profile.get("short_reason", "")
    }
    return normalized


# -----------------------------
# Chunking
# -----------------------------
def _build_background_chunks(user_id: str, structured_profile: Dict, raw_text: str, sources: List[Dict]) -> List[Dict]:
    chunks = []

    def add_chunk(chunk_type: str, text: str, source_type: str = "profile", retrieval_priority: float = 1.0):
        if text and text.strip():
            chunk_id = f"{user_id}_{chunk_type}_{len(chunks)+1:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "text": text.strip(),
                "source_type": source_type,
                "retrieval_priority": retrieval_priority
            })

    role_lens = structured_profile.get("role_lens", "general")
    add_chunk(
        "role_identity",
        f"The user's role lens is {role_lens}. Current role: {structured_profile.get('current_role', role_lens)}.",
        retrieval_priority=1.3
    )

    industry_domain = structured_profile.get("industry_domain", [])
    if industry_domain:
        add_chunk(
            "domain_context",
            "The user works in these domains: " + ", ".join(industry_domain),
            retrieval_priority=1.2
        )

    add_chunk(
        "technical_exposure",
        f"The user's technical depth is {structured_profile.get('technical_depth', 'medium')}.",
        retrieval_priority=1.4
    )

    weak_areas = structured_profile.get("weak_areas", [])
    if weak_areas:
        add_chunk(
            "knowledge_boundary",
            "The user's weaker areas include: " + ", ".join(weak_areas),
            retrieval_priority=1.5
        )

    pref = structured_profile.get("preferred_explanation_style", [])
    if pref:
        add_chunk(
            "expression_preference",
            "The user prefers explanations that are: " + ", ".join(pref),
            retrieval_priority=1.5
        )

    current_projects = structured_profile.get("current_projects", [])
    if current_projects:
        add_chunk(
            "current_project",
            "The user's current projects include: " + "; ".join(current_projects),
            retrieval_priority=1.1
        )

    strength_areas = structured_profile.get("strength_areas", [])
    if strength_areas:
        add_chunk(
            "technical_exposure",
            "The user's stronger areas include: " + ", ".join(strength_areas),
            retrieval_priority=1.0
        )

    jargon_tolerance = structured_profile.get("jargon_tolerance", "medium")
    add_chunk(
        "expression_preference",
        f"The user's jargon tolerance is {jargon_tolerance}.",
        retrieval_priority=1.2
    )

    # Optional raw-text fallback chunk
    if raw_text:
        add_chunk(
            "domain_context",
            raw_text[:700],
            source_type="raw_background",
            retrieval_priority=0.6
        )

    return chunks


# -----------------------------
# Storage
# -----------------------------
def _store_profile(user_id: str, structured_profile: Dict) -> Dict:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO user_profiles (user_id, structured_profile_json)
        VALUES (?, ?)
        """,
        (user_id, json.dumps(structured_profile, ensure_ascii=False))
    )

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "profile_status": "stored",
        "store_type": "sqlite"
    }


def _store_chunks(user_id: str, chunks: List[Dict]) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM background_chunks WHERE user_id = ?", (user_id,))

    for chunk in chunks:
        cur.execute(
            """
            INSERT INTO background_chunks
            (user_id, chunk_id, chunk_type, chunk_text, source_type, retrieval_priority)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                chunk["chunk_id"],
                chunk["chunk_type"],
                chunk["text"],
                chunk.get("source_type", "profile"),
                float(chunk.get("retrieval_priority", 1.0)),
            )
        )

    conn.commit()
    conn.close()

    _rebuild_user_vectors(user_id)

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "vector_store_status": "stored"
        }
        for chunk in chunks
    ]


# -----------------------------
# Public API: onboarding
# -----------------------------
def onboard_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    """
    Person 1 interface 1:
    onboard_user_background(user_id, raw_background_inputs)
        -> structured_profile, background_chunks
    """
    _init_db()

    raw_pkg = _combine_raw_background(raw_background_inputs)
    raw_text = raw_pkg["combined_text"]
    sources = raw_pkg["sources"]

    try:
        parsed = _parse_background_with_llm(raw_text)
        structured_profile = _normalize_profile(parsed)
    except Exception:
        structured_profile = _fallback_profile(raw_text)

    chunks = _build_background_chunks(
        user_id=user_id,
        structured_profile=structured_profile,
        raw_text=raw_text,
        sources=sources
    )

    stored_profile_record = _store_profile(user_id, structured_profile)
    vector_entries = _store_chunks(user_id, chunks)

    return {
        "user_id": user_id,
        "raw_background_package": {
            "user_id": user_id,
            "sources": sources
        },
        "structured_profile": structured_profile,
        "background_chunks": chunks,
        "stored_profile_record": stored_profile_record,
        "vectorized_memory_entries": vector_entries
    }


# -----------------------------
# Retrieval scoring
# -----------------------------
def _score_chunk_for_query(query: str, chunk_meta: Dict, sim_score: float) -> float:
    """
    Final score = semantic similarity + retrieval priority + light lexical bonus
    """
    score = float(sim_score)

    retrieval_priority = float(chunk_meta.get("retrieval_priority", 1.0))
    score += 0.15 * retrieval_priority

    q = query.lower()
    text = chunk_meta.get("chunk_text", "").lower()
    chunk_type = chunk_meta.get("chunk_type", "")

    if chunk_type == "knowledge_boundary" and any(x in q for x in ["how", "explain", "what does", "what is"]):
        score += 0.08

    if chunk_type == "expression_preference" and any(x in q for x in ["explain", "summarize", "understand"]):
        score += 0.08

    if any(token in text for token in q.split()[:6]):
        score += 0.05

    return score


# -----------------------------
# Public API: retrieval
# -----------------------------
def retrieve_user_background(
    user_id: str,
    query: str,
    recommended_chunk_types: List[str],
    top_k: int = 4
) -> Dict:
    """
    Person 1 interface 2:
    retrieve_user_background(user_id, query, recommended_chunk_types)
        -> retrieved_background_package
    """
    _init_db()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        "SELECT structured_profile_json FROM user_profiles WHERE user_id = ?",
        (user_id,)
    )
    row = cur.fetchone()

    structured_profile = {}
    if row and row[0]:
        try:
            structured_profile = json.loads(row[0])
        except Exception:
            structured_profile = {}

    conn.close()

    index, meta = _load_vector_store()

    if index is None or not meta:
        return {
            "user_id": user_id,
            "structured_profile": structured_profile,
            "retrieved_background_chunks": []
        }

    filtered_positions = []
    filtered_meta = []

    for i, m in enumerate(meta):
        if m["user_id"] != user_id:
            continue
        if recommended_chunk_types and m["chunk_type"] not in recommended_chunk_types:
            continue
        filtered_positions.append(i)
        filtered_meta.append(m)

    if not filtered_meta:
        # fallback: ignore chunk type filter, only keep same user
        filtered_positions = []
        filtered_meta = []
        for i, m in enumerate(meta):
            if m["user_id"] == user_id:
                filtered_positions.append(i)
                filtered_meta.append(m)

    if not filtered_meta:
        return {
            "user_id": user_id,
            "structured_profile": structured_profile,
            "retrieved_background_chunks": []
        }

    # Rebuild a tiny temporary FAISS for filtered candidates
    candidate_texts = [m["chunk_text"] for m in filtered_meta]
    candidate_vecs = _embed_texts(candidate_texts)
    dim = candidate_vecs.shape[1]

    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(candidate_vecs)

    qvec = _embed_texts([query])
    k = min(top_k * 3, len(filtered_meta))
    sims, idxs = temp_index.search(qvec, k)

    ranked = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0:
            continue
        m = filtered_meta[int(idx)]
        final_score = _score_chunk_for_query(query, m, float(sim))
        ranked.append({
            "chunk_type": m["chunk_type"],
            "text": m["chunk_text"],
            "score": round(final_score, 4),
            "source_type": m.get("source_type", "profile")
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[:top_k]

    return {
        "user_id": user_id,
        "structured_profile": structured_profile,
        "retrieved_background_chunks": ranked
    }
