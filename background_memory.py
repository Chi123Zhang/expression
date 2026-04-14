import os
import json
from typing import Dict, List, Optional
from openai import OpenAI


PROFILE_DB_PATH = "background_profiles.json"
CHUNK_DB_PATH = "background_chunks.json"


def _ensure_json_file(path: str, default_obj):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str, default_obj):
    _ensure_json_file(path, default_obj)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def parse_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    """
    Parse raw user background into a structured profile.

    raw_background_inputs example:
    [
        {"source_type": "resume", "raw_text": "..."},
        {"source_type": "self_intro", "raw_text": "..."}
    ]
    """
    combined_text = "\n\n".join(
        f"[{item.get('source_type', 'unknown')}]\n{item.get('raw_text', '')}"
        for item in raw_background_inputs
        if item.get("raw_text")
    ).strip()

    if not combined_text:
        raise ValueError("No background text provided.")

    client = _get_openai_client()

    prompt = f"""
You are extracting a structured user background profile for a personalized explanation agent.

Given the user's background text, return ONLY valid JSON with exactly these keys:

- user_id
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
- industry_domain should be a list of strings
- preferred_explanation_style should be a list of strings
- strength_areas should be a list of strings
- weak_areas should be a list of strings
- current_projects should be a list of strings
- short_reason should be a short explanation of the inferred profile

Set user_id to "{user_id}"

Background text:
{combined_text}
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

    profile = json.loads(content)

    valid_role_lens = {"general", "pm", "engineer", "business"}
    valid_depth = {"low", "medium", "high"}

    if profile.get("role_lens") not in valid_role_lens:
        profile["role_lens"] = "general"

    if profile.get("technical_depth") not in valid_depth:
        profile["technical_depth"] = "medium"

    if profile.get("business_depth") not in valid_depth:
        profile["business_depth"] = "medium"

    if profile.get("jargon_tolerance") not in valid_depth:
        profile["jargon_tolerance"] = "medium"

    for key in [
        "industry_domain",
        "preferred_explanation_style",
        "strength_areas",
        "weak_areas",
        "current_projects",
    ]:
        if not isinstance(profile.get(key), list):
            profile[key] = []

    if "short_reason" not in profile:
        profile["short_reason"] = ""

    return profile


def chunk_user_background(raw_background_inputs: List[Dict], structured_profile: Dict) -> List[Dict]:
    """
    Convert background/profile into typed chunks.
    """
    user_id = structured_profile["user_id"]
    chunks = []

    current_role = structured_profile.get("current_role", "")
    role_lens = structured_profile.get("role_lens", "")
    domains = structured_profile.get("industry_domain", [])
    technical_depth = structured_profile.get("technical_depth", "medium")
    business_depth = structured_profile.get("business_depth", "medium")
    preferences = structured_profile.get("preferred_explanation_style", [])
    jargon_tolerance = structured_profile.get("jargon_tolerance", "medium")
    strengths = structured_profile.get("strength_areas", [])
    weak_areas = structured_profile.get("weak_areas", [])
    projects = structured_profile.get("current_projects", [])

    if current_role or role_lens:
        chunks.append({
            "chunk_id": f"{user_id}_role_01",
            "user_id": user_id,
            "chunk_type": "role_identity",
            "text": f"The user currently works as {current_role} and should generally be addressed through a {role_lens} lens."
        })

    if domains:
        chunks.append({
            "chunk_id": f"{user_id}_domain_01",
            "user_id": user_id,
            "chunk_type": "domain_context",
            "text": f"The user's domain background includes: {', '.join(domains)}."
        })

    chunks.append({
        "chunk_id": f"{user_id}_technical_01",
        "user_id": user_id,
        "chunk_type": "technical_exposure",
        "text": f"The user's technical depth is {technical_depth}, business depth is {business_depth}, and jargon tolerance is {jargon_tolerance}."
    })

    if weak_areas:
        chunks.append({
            "chunk_id": f"{user_id}_boundary_01",
            "user_id": user_id,
            "chunk_type": "knowledge_boundary",
            "text": f"The user is less comfortable with: {', '.join(weak_areas)}."
        })

    if preferences:
        chunks.append({
            "chunk_id": f"{user_id}_pref_01",
            "user_id": user_id,
            "chunk_type": "expression_preference",
            "text": f"The user prefers explanations that are: {', '.join(preferences)}."
        })

    if strengths:
        chunks.append({
            "chunk_id": f"{user_id}_strength_01",
            "user_id": user_id,
            "chunk_type": "strength_area",
            "text": f"The user's strength areas include: {', '.join(strengths)}."
        })

    if projects:
        chunks.append({
            "chunk_id": f"{user_id}_project_01",
            "user_id": user_id,
            "chunk_type": "current_project",
            "text": f"The user is currently working on: {', '.join(projects)}."
        })

    raw_text = "\n".join(item.get("raw_text", "") for item in raw_background_inputs if item.get("raw_text")).strip()
    if raw_text:
        chunks.append({
            "chunk_id": f"{user_id}_raw_01",
            "user_id": user_id,
            "chunk_type": "raw_background",
            "text": raw_text[:1500]
        })

    return chunks


def store_profile(user_id: str, structured_profile: Dict) -> Dict:
    db = _load_json(PROFILE_DB_PATH, {})
    db[user_id] = structured_profile
    _save_json(PROFILE_DB_PATH, db)

    return {
        "user_id": user_id,
        "profile_status": "stored",
        "store_type": "json"
    }


def store_chunks(user_id: str, background_chunks: List[Dict]) -> List[Dict]:
    db = _load_json(CHUNK_DB_PATH, {})
    db[user_id] = background_chunks
    _save_json(CHUNK_DB_PATH, db)

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "storage_status": "stored"
        }
        for chunk in background_chunks
    ]


def onboard_user_background(user_id: str, raw_background_inputs: List[Dict]) -> Dict:
    """
    Full onboarding pipeline.
    """
    structured_profile = parse_user_background(user_id, raw_background_inputs)
    background_chunks = chunk_user_background(raw_background_inputs, structured_profile)

    profile_store_result = store_profile(user_id, structured_profile)
    chunk_store_result = store_chunks(user_id, background_chunks)

    return {
        "structured_profile": structured_profile,
        "background_chunks": background_chunks,
        "profile_store_result": profile_store_result,
        "chunk_store_result": chunk_store_result
    }


def _simple_text_score(query: str, text: str) -> int:
    q_terms = set(query.lower().split())
    t_terms = set(text.lower().split())
    return len(q_terms & t_terms)


def retrieve_user_background(
    user_id: str,
    query: str,
    recommended_chunk_types: Optional[List[str]] = None,
    top_k: int = 4
) -> Dict:
    """
    Retrieve user background for downstream personalization.
    Minimal MVP version: structured lookup + simple lexical scoring.
    """
    profiles = _load_json(PROFILE_DB_PATH, {})
    chunks_db = _load_json(CHUNK_DB_PATH, {})

    structured_profile = profiles.get(user_id)
    user_chunks = chunks_db.get(user_id, [])

    if structured_profile is None:
        return {
            "user_id": user_id,
            "structured_profile": None,
            "retrieved_background_chunks": []
        }

    filtered_chunks = user_chunks
    if recommended_chunk_types:
        filtered_chunks = [
            c for c in user_chunks
            if c.get("chunk_type") in recommended_chunk_types
        ]

    scored = []
    for chunk in filtered_chunks:
        score = _simple_text_score(query, chunk["text"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored[:top_k]]

    return {
        "user_id": user_id,
        "structured_profile": {
            "role_lens": structured_profile.get("role_lens", "general"),
            "technical_depth": structured_profile.get("technical_depth", "medium"),
            "business_depth": structured_profile.get("business_depth", "medium"),
            "jargon_tolerance": structured_profile.get("jargon_tolerance", "medium"),
            "preferred_explanation_style": structured_profile.get("preferred_explanation_style", []),
            "short_reason": structured_profile.get("short_reason", "")
        },
        "retrieved_background_chunks": top_chunks
    }
