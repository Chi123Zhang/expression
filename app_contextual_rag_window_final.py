import os
import re
import json
import tempfile
import zipfile
from io import BytesIO

import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import requests
from bs4 import BeautifulSoup

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background


st.set_page_config(page_title="ENVIO LLM Coding Assistant", layout="wide")
st.title("ENVIO LLM Coding Assistant")
st.caption("Document-grounded LLM coding, topic modeling, and LLM-human coding comparison for ENVIO transcripts")


CODEBOOK = [
    "environmental_barrier",
    "social_support",
    "healthcare_access",
    "stigma",
    "mental_health",
]

DEFAULT_CODING_QUERY = (
    "Please code the uploaded de-identified qualitative transcript using the ENVIO codebook."
)

BATCH_SIZE = 5
RAG_CONTEXT_CHAR_LIMIT = 2000


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def infer_participant_id(names):
    joined = " ".join(names)
    match = re.search(r"\b(\d{3}[A-Za-z]?)\b", joined)
    if match:
        return match.group(1)
    return safe_filename(os.path.splitext(names[0])[0][:40])


def infer_date_from_name(name):
    match = re.search(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})", str(name))
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return ""


def infer_source_type(name, text=""):
    low = f"{name} {text[:1000]}".lower()
    if any(x in low for x in ["costing", "cost questions", "cost-effectiveness", "cost "]):
        return "costing"
    if any(x in low for x in ["policy", "protocol", "irb", "datasheet", "human subjects", "privacy", "security"]):
        return "policy"
    if any(x in low for x in ["interview", "qual interview", "transcript", "participant", "moderator"]):
        return "interview"
    return "other"


def chunk_text(text: str, chunk_chars: int = 3500, overlap_chars: int = 300):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def batch_list(items, batch_size=BATCH_SIZE):
    """Split a list into small batches for fewer, more consistent API calls."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def build_contextual_retrieval_query(chunk_batches, batch_index, window_size=1, max_chars=1800):
    """Build a local-context retrieval query for RAG.

    Coding still happens only for the current batch, preserving one output row
    per human-aligned segment. Retrieval, however, sees the previous and next
    neighboring batches so that ambiguous segments are interpreted with local
    transcript context.
    """
    start = max(0, batch_index - window_size)
    end = min(len(chunk_batches), batch_index + window_size + 1)

    blocks = []
    for j in range(start, end):
        label = "CURRENT BATCH" if j == batch_index else (
            "PREVIOUS CONTEXT" if j < batch_index else "NEXT CONTEXT"
        )
        batch_text = "\n".join(chunk_batches[j])
        blocks.append(f"[{label}]\n{batch_text}")

    query = "\n\n".join(blocks).strip()
    return query[:max_chars]


def load_transcript_text(uploaded_file) -> str:
    uploaded_file.seek(0)
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            pages = load_pdf(tmp_path)
            return "\n".join(page_text for _, page_text in pages)
        if suffix == ".docx":
            pages = load_docx(tmp_path)
            return "\n".join(page_text for _, page_text in pages)
        if suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()

        if url.lower().endswith(".pdf") or "application/pdf" in content_type:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name
            try:
                pages = load_pdf(tmp_path)
                return "\n".join(page_text for _, page_text in pages)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)

    except Exception as e:
        st.warning(f"Could not load URL: {url}. Reason: {e}")
        return ""


def build_source_records(uploaded_files, url_input, merge_files=True):
    source_records = []

    if uploaded_files:
        for f in uploaded_files:
            text = load_transcript_text(f)
            if text.strip():
                source_records.append({
                    "source_name": f.name,
                    "source_text": text,
                    "source_kind": infer_source_type(f.name, text),
                    "source_date": infer_date_from_name(f.name),
                })

    if url_input.strip():
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        for url in urls:
            text = load_text_from_url(url)
            if text.strip():
                source_records.append({
                    "source_name": url,
                    "source_text": text,
                    "source_kind": infer_source_type(url, text),
                    "source_date": infer_date_from_name(url),
                })

    if not source_records:
        return []

    if merge_files:
        participant_id = infer_participant_id([r["source_name"] for r in source_records])
        for r in source_records:
            r["participant_id"] = participant_id
        return source_records

    for r in source_records:
        r["participant_id"] = infer_participant_id([r["source_name"]])
    return source_records


def extract_json_object(content: str):
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return {"segments": parsed}
        return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("Could not parse JSON", content, 0)


def call_openai_json(client, prompt, system_message="You are a careful qualitative coding assistant. Return strict JSON only."):
    """Call OpenAI and parse a strict JSON response."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    return extract_json_object(content)


def normalize_segments(parsed, fallback_text, source_section="", source_kind=""):
    """Normalize model JSON into a list of segment dictionaries."""
    if isinstance(parsed, list):
        segments = parsed
    elif isinstance(parsed, dict):
        segments = parsed.get("segments", [])
    else:
        segments = []

    if not segments:
        return [{
            "text": fallback_text,
            "codes": [],
            "rationale": "No codable segment was returned.",
            "source_section": source_section,
            "source_type": source_kind,
        }]

    cleaned = []
    allowed = set(CODEBOOK)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        codes = [c for c in clean_code_list(seg.get("codes", [])) if c in allowed]
        cleaned.append({
            "text": str(seg.get("text", "")).strip() or fallback_text[:1000],
            "codes": codes,
            "rationale": str(seg.get("rationale", "")).strip(),
            "source_section": seg.get("source_section", source_section),
            "source_type": seg.get("source_type", source_kind),
        })

    return cleaned or [{
        "text": fallback_text,
        "codes": [],
        "rationale": "No valid segment was returned after normalization.",
        "source_section": source_section,
        "source_type": source_kind,
    }]


def run_initial_llm_coding(text_segment, codebook, client, source_section="", source_kind=""):
    """
    Stage 1: pure transcript coding without RAG.
    This is also used as the No-RAG baseline in the paired comparison.
    """
    prompt = f"""
You are an expert qualitative research coding assistant for the ENVIO / TechMPower study.

Research goal:
Compare LLM-generated qualitative coding against human coding for de-identified qualitative transcripts.
The focus is how environmental, institutional, social, mental health, stigma, and healthcare factors influence HIV care engagement.

Source type:
{source_kind}

Source section:
{source_section}

Allowed codebook:
{codebook}

Transcript or document segment:
{text_segment}

Instructions:
1. Split the segment into meaningful qualitative units.
2. Assign zero, one, or multiple codes from the allowed codebook only.
3. Do not invent codes outside the codebook.
4. Preserve short evidence excerpts.
5. If the segment is policy/protocol material, code only if it clearly relates to the codebook.
6. Output ONLY valid JSON. No markdown. No explanation.

Required JSON schema:
{{
  "segments": [
    {{
      "text": "meaningful unit",
      "codes": ["code1", "code2"],
      "rationale": "brief reason grounded in the transcript text",
      "source_section": "{source_section}",
      "source_type": "{source_kind}"
    }}
  ]
}}
"""
    parsed = call_openai_json(client, prompt)
    return normalize_segments(parsed, text_segment, source_section, source_kind)


def normalize_batch_segments(parsed, segment_batch, source_section="", source_kind=""):
    """Normalize a batch JSON response into exactly one output row per input segment.

    The model is asked to return input_id=1..N. If it fails to do so, we fall back
    to sequential order. This preserves human CSV alignment: one human row -> one
    LLM row.
    """
    if isinstance(parsed, list):
        raw_segments = parsed
    elif isinstance(parsed, dict):
        raw_segments = parsed.get("segments", [])
    else:
        raw_segments = []

    by_id = {}
    sequential = []
    allowed = set(CODEBOOK)

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        raw_id = seg.get("input_id", seg.get("segment_id", seg.get("id", None)))
        try:
            input_id = int(raw_id)
        except Exception:
            input_id = None

        codes = [c for c in clean_code_list(seg.get("codes", [])) if c in allowed]
        rationale = str(seg.get("rationale", "")).strip()

        clean_seg = {
            "codes": codes,
            "rationale": rationale,
        }

        if input_id is not None and 1 <= input_id <= len(segment_batch):
            if input_id not in by_id:
                by_id[input_id] = {"codes": [], "rationales": []}
            for c in codes:
                if c not in by_id[input_id]["codes"]:
                    by_id[input_id]["codes"].append(c)
            if rationale:
                by_id[input_id]["rationales"].append(rationale)
        else:
            sequential.append(clean_seg)

    output = []
    for idx, original_text in enumerate(segment_batch, start=1):
        if idx in by_id:
            codes = by_id[idx]["codes"]
            rationale = " | ".join(by_id[idx]["rationales"])[:1200]
        elif idx - 1 < len(sequential):
            codes = sequential[idx - 1]["codes"]
            rationale = sequential[idx - 1]["rationale"][:1200]
        else:
            codes = []
            rationale = "No coding returned for this segment in the batch response."

        output.append({
            "text": original_text,
            "codes": codes,
            "rationale": rationale,
            "source_section": source_section,
            "source_type": source_kind,
        })

    return output


def run_initial_llm_batch_coding(segment_batch, codebook, client, source_section="", source_kind=""):
    """Stage 1 batch coding: several already-aligned segments in one API call."""
    formatted_segments = []
    for i, seg in enumerate(segment_batch, start=1):
        formatted_segments.append(f"INPUT_ID {i}:\n{seg}")
    joined_segments = "\n\n".join(formatted_segments)

    prompt = f"""
You are an expert qualitative research coding assistant for the ENVIO / TechMPower study.

Research goal:
Compare LLM-generated qualitative coding against human coding for de-identified qualitative transcripts.
The focus is how environmental, institutional, social, mental health, stigma, and healthcare factors influence HIV care engagement.

Allowed codebook:
{codebook}

Segments to code:
{joined_segments}

Instructions:
1. Code each INPUT_ID independently.
2. Return exactly one JSON object per INPUT_ID.
3. Do not merge different INPUT_IDs.
4. Do not split one INPUT_ID into multiple output rows.
5. Assign zero, one, or multiple codes from the allowed codebook only.
6. Do not invent codes outside the codebook.
7. Preserve the original text as much as possible.
8. Output ONLY valid JSON. No markdown. No explanation outside JSON.

Required JSON schema:
{{
  "segments": [
    {{
      "input_id": 1,
      "text": "original segment text",
      "codes": ["code1", "code2"],
      "rationale": "brief reason grounded in the transcript text",
      "source_section": "{source_section}",
      "source_type": "{source_kind}"
    }}
  ]
}}
"""
    parsed = call_openai_json(client, prompt)
    return normalize_batch_segments(parsed, segment_batch, source_section, source_kind)


def run_rag_batch_refinement_coding(initial_segments, segment_batch, codebook, client, retrieved_context="", source_section="", source_kind=""):
    """Stage 2 batch RAG refinement while preserving one row per input segment."""
    if not retrieved_context or not str(retrieved_context).strip():
        return initial_segments

    initial_with_ids = []
    for i, seg in enumerate(initial_segments, start=1):
        item = dict(seg)
        item["input_id"] = i
        initial_with_ids.append(item)

    formatted_segments = []
    for i, seg in enumerate(segment_batch, start=1):
        formatted_segments.append(f"INPUT_ID {i}:\n{seg}")

    initial_json = json.dumps({"segments": initial_with_ids}, ensure_ascii=False, indent=2)
    joined_segments = "\n\n".join(formatted_segments)

    prompt = f"""
You are refining qualitative coding for the ENVIO / TechMPower study.

Important principle:
The transcript text is the primary evidence. The RAG context is supporting project/codebook context.
Use RAG to clarify code definitions, study framing, and borderline cases. Do not add codes that are not supported by the transcript segment.

Allowed codebook:
{codebook}

Original input segments:
{joined_segments}

Initial No-RAG coding result:
{initial_json}

Relevant RAG context from project documents:
{retrieved_context[:RAG_CONTEXT_CHAR_LIMIT]}

Refinement instructions:
1. Review each INPUT_ID independently.
2. Return exactly one JSON object per INPUT_ID.
3. Preserve INPUT_ID values and order.
4. Keep the initial coding if it is already well supported.
5. Revise codes only when the RAG context clearly improves consistency with the codebook or study framing.
6. You may add, remove, or replace codes, but only using the allowed codebook.
7. Preserve the original segment text as much as possible.
8. Add a short rationale explaining whether the code was kept or refined.
9. Output ONLY valid JSON. No markdown. No explanation outside JSON.

Required JSON schema:
{{
  "segments": [
    {{
      "input_id": 1,
      "text": "original segment text",
      "codes": ["code1", "code2"],
      "rationale": "kept/refined reason grounded in transcript + RAG context",
      "source_section": "{source_section}",
      "source_type": "{source_kind}"
    }}
  ]
}}
"""
    parsed = call_openai_json(
        client,
        prompt,
        system_message="You are a careful qualitative coding refinement assistant. Return strict JSON only.",
    )
    return normalize_batch_segments(parsed, segment_batch, source_section, source_kind)


def run_rag_refinement_coding(initial_segments, text_segment, codebook, client, retrieved_context="", source_section="", source_kind=""):
    """
    Stage 2: RAG refinement.
    The model reviews the initial No-RAG coding against retrieved project/codebook context
    and only changes codes when the evidence supports the correction.
    """
    if not retrieved_context or not str(retrieved_context).strip():
        return initial_segments

    initial_json = json.dumps({"segments": initial_segments}, ensure_ascii=False, indent=2)

    prompt = f"""
You are refining qualitative coding for the ENVIO / TechMPower study.

Important principle:
The transcript text is the primary evidence. The RAG context is supporting project/codebook context.
Use RAG to clarify code definitions, study framing, and borderline cases. Do not add codes that are not supported by the transcript segment.

Allowed codebook:
{codebook}

Original transcript or document segment:
{text_segment}

Initial No-RAG coding result:
{initial_json}

Relevant RAG context from project documents:
{retrieved_context[:RAG_CONTEXT_CHAR_LIMIT]}

Refinement instructions:
1. Review each initial coded segment.
2. Keep the initial coding if it is already well supported.
3. Revise codes only when the RAG context clearly improves consistency with the codebook or study framing.
4. You may add, remove, or replace codes, but only using the allowed codebook.
5. Preserve the original segment text as much as possible.
6. Add a short rationale explaining whether the code was kept or refined.
7. Output ONLY valid JSON. No markdown. No explanation outside JSON.

Required JSON schema:
{{
  "segments": [
    {{
      "text": "meaningful unit",
      "codes": ["code1", "code2"],
      "rationale": "kept/refined reason grounded in transcript + RAG context",
      "source_section": "{source_section}",
      "source_type": "{source_kind}"
    }}
  ]
}}
"""
    parsed = call_openai_json(
        client,
        prompt,
        system_message="You are a careful qualitative coding refinement assistant. Return strict JSON only.",
    )
    return normalize_segments(parsed, text_segment, source_section, source_kind)


def run_llm_coding_with_context(text_segment, codebook, client, retrieved_context="", source_section="", source_kind=""):
    """
    Two-stage coding pipeline:
    1) No-RAG initial coding.
    2) Optional RAG-based refinement.

    If retrieved_context is empty, this returns the Stage 1 No-RAG result.
    If retrieved_context is provided, this returns the refined RAG result.
    """
    try:
        initial_segments = run_initial_llm_coding(
            text_segment=text_segment,
            codebook=codebook,
            client=client,
            source_section=source_section,
            source_kind=source_kind,
        )

        refined_segments = run_rag_refinement_coding(
            initial_segments=initial_segments,
            text_segment=text_segment,
            codebook=codebook,
            client=client,
            retrieved_context=retrieved_context,
            source_section=source_section,
            source_kind=source_kind,
        )
        return refined_segments

    except Exception as e:
        return [{
            "text": text_segment,
            "codes": ["PARSE_ERROR"],
            "rationale": f"The coding/refinement output could not be completed: {e}",
            "source_section": source_section,
            "source_type": source_kind,
        }]

def get_rag_context(rag, query, profile, use_rag=True):
    if not use_rag:
        return ""
    try:
        result = rag.answer_question(
            query=query,
            mode="coding",
            role=profile["role"],
            user_profile=profile,
        )
        return result.get("retrieved_context", "")
    except Exception as e:
        st.warning(f"RAG retrieval skipped for one chunk: {e}")
        return ""


def clean_code_list(codes):
    if isinstance(codes, list):
        raw = codes
    else:
        raw = str(codes).replace(";", ",").split(",")

    cleaned = []
    for c in raw:
        c = str(c).strip()
        if c and c.lower() not in ["nan", "none", "null", "[]"]:
            cleaned.append(c)
    return cleaned


def make_coding_dataframe(segments, participant_id, source_name, source_kind, source_date):
    rows = []
    for i, seg in enumerate(segments, start=1):
        codes = clean_code_list(seg.get("codes", []))
        rows.append({
            "participant_id": participant_id,
            "segment_index": i,
            "source_name": source_name,
            "source_type": seg.get("source_type", source_kind),
            "source_date": source_date,
            "source_section": seg.get("source_section", ""),
            "text": seg.get("text", ""),
            "codes": ",".join(codes),
            "rationale": seg.get("rationale", ""),
        })
    return pd.DataFrame(rows)

def build_human_aligned_chunks(human_df, participant_id=None):
    """Use the human coding CSV as the segmentation template.

    This prevents the LLM from creating a different number of segments than the
    human benchmark. When a human coding CSV is uploaded, each human row becomes
    one LLM coding unit, so segment_index aligns exactly for F1/Kappa.
    """
    if human_df is None or human_df.empty or "text" not in human_df.columns:
        return []

    df = human_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "segment_index" not in df.columns and "segment_id" in df.columns:
        df = df.rename(columns={"segment_id": "segment_index"})

    if participant_id and "participant_id" in df.columns:
        pid = str(participant_id).strip()
        filtered = df[df["participant_id"].astype(str).str.strip() == pid]
        if not filtered.empty:
            df = filtered

    if "segment_index" in df.columns:
        df["segment_index"] = pd.to_numeric(df["segment_index"], errors="coerce")
        df = df.dropna(subset=["segment_index"]).sort_values("segment_index")

    chunks = []
    for _, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if text:
            chunks.append(text)
    return chunks


def collapse_to_one_human_aligned_segment(coded_segments, original_text, source_section, source_kind):
    """Collapse any model-produced subsegments back to one row.

    Even if the model tries to split text internally, evaluation stays aligned
    with the human CSV: one input human row -> one output LLM row.
    """
    allowed = set(CODEBOOK)
    codes = []
    rationales = []
    for seg in coded_segments or []:
        for c in clean_code_list(seg.get("codes", [])):
            if c in allowed and c not in codes:
                codes.append(c)
        r = str(seg.get("rationale", "")).strip()
        if r:
            rationales.append(r)

    return [{
        "text": original_text,
        "codes": codes,
        "rationale": " | ".join(rationales)[:1200],
        "source_section": source_section,
        "source_type": source_kind,
    }]


def code_frequency(df):
    if df.empty or "codes" not in df.columns:
        return pd.Series(dtype=int)

    clean_codes = (
        df["codes"]
        .fillna("")
        .apply(lambda x: ",".join(clean_code_list(x)))
    )

    counts = clean_codes.str.get_dummies(sep=",").sum().sort_values(ascending=False)
    counts = counts[counts.index.astype(str).str.strip() != ""]
    return counts


def code_frequency_by_group(df, group_col):
    if df.empty or group_col not in df.columns or "codes" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["codes"] = tmp["codes"].fillna("").apply(lambda x: ",".join(clean_code_list(x)))
    dummies = tmp["codes"].str.get_dummies(sep=",")
    dummies = dummies.loc[:, [c for c in dummies.columns if str(c).strip() != ""]]

    if dummies.empty:
        return pd.DataFrame()

    return pd.concat([tmp[[group_col]], dummies], axis=1).groupby(group_col).sum()


def normalize_code_string(x):
    return clean_code_list(x)

def compare_llm_human(llm_df, human_df, codebook=None):
    results = []

    llm_df = llm_df.copy()
    human_df = human_df.copy()

    llm_df.columns = [str(c).strip() for c in llm_df.columns]
    human_df.columns = [str(c).strip() for c in human_df.columns]

    if "segment_index" not in human_df.columns and "segment_id" in human_df.columns:
        human_df = human_df.rename(columns={"segment_id": "segment_index"})

    if "segment_index" not in llm_df.columns:
        llm_df["segment_index"] = range(1, len(llm_df) + 1)

    if "segment_index" not in human_df.columns:
        human_df["segment_index"] = range(1, len(human_df) + 1)

    llm_df["segment_index"] = pd.to_numeric(llm_df["segment_index"], errors="coerce")
    human_df["segment_index"] = pd.to_numeric(human_df["segment_index"], errors="coerce")

    llm_df = llm_df.dropna(subset=["segment_index"])
    human_df = human_df.dropna(subset=["segment_index"])

    llm_df["segment_index"] = llm_df["segment_index"].astype(int)
    human_df["segment_index"] = human_df["segment_index"].astype(int)

    metadata_cols = {
        "participant_id", "segment_index", "segment_id", "source_name",
        "source_type", "source_date", "source_section", "text",
        "codes", "rationale"
    }

    # Codes from LLM string column
    llm_codes_found = set()
    if "codes" in llm_df.columns:
        for x in llm_df["codes"].fillna(""):
            llm_codes_found.update(clean_code_list(x))

    # Codes from human binary columns
    human_code_cols = {
        c for c in human_df.columns
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(human_df[c])
    }

    # Optional codebook + observed codes
    if codebook is None:
        codebook = []

    dynamic_codebook = sorted(
        set(codebook).union(llm_codes_found).union(human_code_cols)
    )

    dynamic_codebook = [
        c for c in dynamic_codebook
        if str(c).strip() and c not in ["ERROR", "PARSE_ERROR"]
    ]

    if not dynamic_codebook:
        return pd.DataFrame([{
            "code": "NO_CODES_FOUND",
            "cohen_kappa": None,
            "f1": None,
            "human_positive": 0,
            "llm_positive": 0,
            "matched_segments": 0,
        }])

    # LLM binary
    llm_binary = llm_df[["segment_index"]].copy()
    for code in dynamic_codebook:
        if "codes" in llm_df.columns:
            llm_binary[code] = llm_df["codes"].fillna("").apply(
                lambda x: int(code in clean_code_list(x))
            )
        else:
            llm_binary[code] = 0

    # Human binary
    human_binary = human_df[["segment_index"]].copy()
    if "codes" in human_df.columns:
        for code in dynamic_codebook:
            human_binary[code] = human_df["codes"].fillna("").apply(
                lambda x: int(code in clean_code_list(x))
            )
    else:
        for code in dynamic_codebook:
            if code in human_df.columns:
                human_binary[code] = pd.to_numeric(
                    human_df[code], errors="coerce"
                ).fillna(0).astype(int).clip(0, 1)
            else:
                human_binary[code] = 0

    merged = pd.merge(
        llm_binary,
        human_binary,
        on="segment_index",
        suffixes=("_llm", "_human"),
        how="inner",
    )

    if merged.empty:
        n = min(len(llm_binary), len(human_binary))
        merged = pd.concat(
            [
                llm_binary[dynamic_codebook].head(n).reset_index(drop=True).add_suffix("_llm"),
                human_binary[dynamic_codebook].head(n).reset_index(drop=True).add_suffix("_human"),
            ],
            axis=1,
        )

    if merged.empty:
        return pd.DataFrame([{
            "code": "NO_MATCH",
            "cohen_kappa": None,
            "f1": None,
            "human_positive": 0,
            "llm_positive": 0,
            "matched_segments": 0,
        }])

    for code in dynamic_codebook:
        y_llm = pd.to_numeric(
            merged[f"{code}_llm"], errors="coerce"
        ).fillna(0).astype(int)

        y_human = pd.to_numeric(
            merged[f"{code}_human"], errors="coerce"
        ).fillna(0).astype(int)

        try:
            kappa = round(cohen_kappa_score(y_human, y_llm), 4)
        except Exception:
            kappa = None

        try:
            f1 = round(f1_score(y_human, y_llm, zero_division=0), 4)
        except Exception:
            f1 = None

        results.append({
            "code": code,
            "cohen_kappa": kappa,
            "f1": f1,
            "human_positive": int(y_human.sum()),
            "llm_positive": int(y_llm.sum()),
            "matched_segments": len(merged),
        })

    macro_f1 = pd.Series([r["f1"] for r in results if r["f1"] is not None]).mean()
    macro_kappa = pd.Series([r["cohen_kappa"] for r in results if r["cohen_kappa"] is not None]).mean()

    results.append({
        "code": "MACRO_AVERAGE",
        "cohen_kappa": round(macro_kappa, 4) if pd.notna(macro_kappa) else None,
        "f1": round(macro_f1, 4) if pd.notna(macro_f1) else None,
        "human_positive": "",
        "llm_positive": "",
        "matched_segments": len(merged),
    })

    return pd.DataFrame(results)


def extract_macro_metrics(comparison_df):
    """Extract macro-level F1 and Cohen's Kappa from a comparison table."""
    if comparison_df is None or comparison_df.empty:
        return {"macro_f1": None, "macro_kappa": None}

    macro = comparison_df[comparison_df["code"] == "MACRO_AVERAGE"]
    if macro.empty:
        return {"macro_f1": None, "macro_kappa": None}

    row = macro.iloc[0]
    return {
        "macro_f1": row.get("f1", None),
        "macro_kappa": row.get("cohen_kappa", None),
    }


def build_rag_comparison_table(no_rag_comparison, rag_comparison):
    """Build a research-style macro comparison table for No-RAG vs RAG coding."""
    no_rag_metrics = extract_macro_metrics(no_rag_comparison)
    rag_metrics = extract_macro_metrics(rag_comparison)

    table = pd.DataFrame([
        {
            "setting": "No RAG",
            "macro_f1": no_rag_metrics["macro_f1"],
            "macro_kappa": no_rag_metrics["macro_kappa"],
        },
        {
            "setting": "RAG",
            "macro_f1": rag_metrics["macro_f1"],
            "macro_kappa": rag_metrics["macro_kappa"],
        },
    ])

    try:
        table["delta_macro_f1_vs_no_rag"] = table["macro_f1"] - table.loc[0, "macro_f1"]
        table["delta_macro_kappa_vs_no_rag"] = table["macro_kappa"] - table.loc[0, "macro_kappa"]
    except Exception:
        table["delta_macro_f1_vs_no_rag"] = None
        table["delta_macro_kappa_vs_no_rag"] = None

    return table
    
# def compare_llm_human(llm_df, human_df, codebook):
#     results = []

#     join_cols = []
#     for col in ["participant_id", "source_type", "segment_index"]:
#         if col in llm_df.columns and col in human_df.columns:
#             join_cols.append(col)

#     if join_cols:
#         merged = pd.merge(
#             llm_df,
#             human_df,
#             on=join_cols,
#             suffixes=("_llm", "_human"),
#             how="inner",
#         )
#         llm_col = "codes_llm"
#         human_col = "codes_human"
#     else:
#         n = min(len(llm_df), len(human_df))
#         merged = pd.DataFrame({
#             "codes_llm": llm_df["codes"].head(n).values,
#             "codes_human": human_df["codes"].head(n).values,
#         })
#         llm_col = "codes_llm"
#         human_col = "codes_human"

#     if merged.empty:
#         return pd.DataFrame([{
#             "code": "NO_MATCH",
#             "cohen_kappa": None,
#             "f1": None,
#             "human_positive": 0,
#             "llm_positive": 0,
#             "matched_segments": 0,
#         }])

#     for code in codebook:
#         y_llm = merged[llm_col].apply(lambda x: int(code in normalize_code_string(x)))
#         y_human = merged[human_col].apply(lambda x: int(code in normalize_code_string(x)))

#         try:
#             kappa = round(cohen_kappa_score(y_human, y_llm), 4)
#         except Exception:
#             kappa = None

#         try:
#             f1 = round(f1_score(y_human, y_llm, zero_division=0), 4)
#         except Exception:
#             f1 = None

#         results.append({
#             "code": code,
#             "cohen_kappa": kappa,
#             "f1": f1,
#             "human_positive": int(y_human.sum()),
#             "llm_positive": int(y_llm.sum()),
#             "matched_segments": len(merged),
#         })

#     macro_f1 = pd.Series([r["f1"] for r in results if r["f1"] is not None]).mean()
#     macro_kappa = pd.Series([r["cohen_kappa"] for r in results if r["cohen_kappa"] is not None]).mean()

#     results.append({
#         "code": "MACRO_AVERAGE",
#         "cohen_kappa": round(macro_kappa, 4) if pd.notna(macro_kappa) else None,
#         "f1": round(macro_f1, 4) if pd.notna(macro_f1) else None,
#         "human_positive": "",
#         "llm_positive": "",
#         "matched_segments": len(merged),
#     })

#     return pd.DataFrame(results)


def run_lda_topic_modeling(texts, n_topics=5, n_words=10):
    custom_stop_words = [
        "like", "think", "know", "maybe", "kind", "sort", "just", "yeah",
        "um", "uh", "really", "thing", "things", "people", "lot", "don",
        "does", "did", "going", "say", "said", "way", "got", "get",
        "okay", "right", "actually", "probably", "basically", "stuff"
    ]

    texts = [str(t).lower() for t in texts if str(t).strip()]
    if len(texts) < 3:
        return pd.DataFrame(), pd.DataFrame()

    english_stop = CountVectorizer(stop_words="english").get_stop_words()
    stop_words = list(set(english_stop).union(custom_stop_words))

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_df=0.85,
        min_df=1,
        max_features=1500,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
    )

    doc_topic = lda.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()

    topic_rows = []
    for topic_idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[-n_words:][::-1]
        words = [feature_names[i] for i in top_idx]
        topic_rows.append({
            "topic_id": topic_idx,
            "top_words": ", ".join(words),
        })

    doc_topic_df = pd.DataFrame(doc_topic, columns=[f"topic_{i}" for i in range(n_topics)])
    doc_topic_df["dominant_topic"] = doc_topic_df[[f"topic_{i}" for i in range(n_topics)]].idxmax(axis=1)

    return pd.DataFrame(topic_rows), doc_topic_df


def run_bertopic_optional(texts):
    try:
        from bertopic import BERTopic
    except Exception:
        return None, "BERTopic is not installed. Add bertopic to requirements.txt if needed."

    texts = [str(t) for t in texts if str(t).strip()]
    if len(texts) < 5:
        return None, "BERTopic needs more text segments."

    try:
        model = BERTopic(verbose=False)
        topics, probs = model.fit_transform(texts)
        info = model.get_topic_info()
        return info, None
    except Exception as e:
        return None, str(e)


def report_safe_text(x, max_len=1200):
    text = str(x).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text[:max_len]


def generate_pdf_report(summary_df, code_counts, group_counts=None, topic_df=None, comparison_df=None, output_file="report.pdf"):
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("ENVIO LLM Coding Summary Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Total coded segments: {len(summary_df)}", styles["Normal"]))
    story.append(Paragraph(f"Participants represented: {summary_df['participant_id'].nunique()}", styles["Normal"]))
    story.append(Paragraph(f"Source types: {', '.join(sorted(summary_df['source_type'].dropna().unique()))}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overall Code Frequencies", styles["Heading2"]))
    for code, count in code_counts.items():
        story.append(Paragraph(report_safe_text(f"{code}: {int(count)}"), styles["Normal"]))

    if group_counts is not None and not group_counts.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Grouped Code Frequencies by Source Type", styles["Heading2"]))
        for source_type, row in group_counts.iterrows():
            pairs = "; ".join([f"{str(c)}={int(v)}" for c, v in row.items()])
            story.append(Paragraph(report_safe_text(f"{source_type}: {pairs}", 1000), styles["Normal"]))

    if topic_df is not None and not topic_df.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Topic Modeling Results", styles["Heading2"]))
        for _, row in topic_df.iterrows():
            story.append(Paragraph(report_safe_text(f"Topic {row['topic_id']}: {row['top_words']}", 800), styles["Normal"]))

    if comparison_df is not None and not comparison_df.empty:
        story.append(Spacer(1, 12))
        story.append(Paragraph("LLM vs Human Coding Agreement", styles["Heading2"]))
        for _, row in comparison_df.iterrows():
            line = (
                f"{row['code']} | Kappa={row['cohen_kappa']} | F1={row['f1']} | "
                f"Human={row['human_positive']} | LLM={row['llm_positive']}"
            )
            story.append(Paragraph(report_safe_text(line, 500), styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Sample Coded Segments", styles["Heading2"]))
    for _, row in summary_df.head(5).iterrows():
        sample = f"Codes: {row['codes']} | Source: {row['source_type']} | Text: {str(row['text'])[:500]}"
        story.append(Paragraph(report_safe_text(sample, 900), styles["Normal"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    return output_file


def create_zip_from_files(file_paths):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            if os.path.exists(path):
                zf.write(path, arcname=os.path.basename(path))
    buffer.seek(0)
    return buffer


@st.cache_resource(show_spinner=False)
def get_cached_rag():
    """Load the RAG system once per Streamlit Cloud session.

    This avoids rebuilding/loading the embedding index every time the app reruns.
    The index should be loaded from techmpower_index when it already exists.
    """
    return initialize_rag(docs_dir=".", force_rebuild=False)


def load_rag_if_needed():
    """Lazy-load RAG only when a mode actually needs retrieval."""
    with st.spinner("Loading RAG system only when needed..."):
        return get_cached_rag()


st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Choose mode", ["coding", "qa", "summary"])
manual_role = st.sidebar.selectbox("Choose response perspective", ["general", "pm", "engineer", "business"])

uploaded_files = st.sidebar.file_uploader(
    "Upload transcript/document files (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

url_input = st.sidebar.text_area(
    "Optional: enter URLs, one per line",
    placeholder="https://example.com/file.pdf\nhttps://example.com/page",
)

merge_related_files = st.sidebar.checkbox(
    "Merge uploaded files/URLs as one participant/interview",
    value=True,
)

use_rag_context = st.sidebar.checkbox("Use RAG context for coding", value=True)
run_rag_comparison = st.sidebar.checkbox(
    "Run paired RAG vs No-RAG comparison",
    value=False,
    help=(
        "Run the same coding pipeline twice: once without RAG and once with RAG. "
        "If a human coding CSV is uploaded, the app will compare both settings "
        "against human coding using Macro F1 and Cohen's Kappa."
    ),
)
st.sidebar.info(f"Batch coding enabled: {BATCH_SIZE} segments per API call")
show_context = st.sidebar.checkbox("Show retrieved context sample", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=True)

human_coding_file = st.sidebar.file_uploader(
    "Optional: upload human coding CSV for comparison",
    type=["csv"],
)

use_human_segments_for_alignment = st.sidebar.checkbox(
    "Use human CSV text as segment template",
    value=True,
    help=(
        "Recommended when comparing with human coding. This makes each human CSV row "
        "one LLM coding unit so segment_index, text, F1, and Kappa are aligned."
    ),
)

run_lda = st.sidebar.checkbox("Run LDA topic modeling", value=True)
n_topics = st.sidebar.slider("Number of LDA topics", min_value=2, max_value=10, value=5)
run_bertopic = st.sidebar.checkbox("Try BERTopic if installed", value=False)

use_resume_profile = st.sidebar.checkbox("Use uploaded transcript to infer profile", value=True)
allow_manual_override = st.sidebar.checkbox("Allow manual role override", value=True)
user_id = st.sidebar.text_input("User ID", value="demo_user")

query = st.text_area("Enter your question or coding instruction (optional)", height=120)


if st.button("Run"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # Lazy RAG: do not load sentence-transformer / FAISS at startup.
    # Load only when coding uses RAG, paired comparison is enabled, or QA/Summary mode is used.
    rag = None

    user_output_dir = os.path.join("outputs", safe_filename(user_id))
    os.makedirs(user_output_dir, exist_ok=True)

    if "users" not in st.session_state:
        st.session_state["users"] = {}

    if user_id not in st.session_state.users:
        st.session_state.users[user_id] = {"profile": {}, "outputs": []}

    profile = {
        "role": manual_role,
        "technical_level": "medium",
        "goal": "understanding",
        "short_reason": "",
    }

    task_query = query.strip() or DEFAULT_CODING_QUERY

    if mode == "coding":
        source_records = build_source_records(
            uploaded_files=uploaded_files,
            url_input=url_input,
            merge_files=merge_related_files,
        )

        if not source_records:
            st.warning("Please upload files or enter at least one URL.")
            st.stop()

        human_df_for_alignment = None
        if human_coding_file is not None:
            try:
                human_coding_file.seek(0)
                human_df_for_alignment = pd.read_csv(human_coding_file)
            except Exception as e:
                st.warning(f"Could not read human coding CSV for alignment: {e}")

        if use_resume_profile:
            try:
                profile_text = "\n".join([r["source_text"][:3000] for r in source_records])[:8000]
                onboard_user_background(
                    user_id=user_id,
                    raw_background_inputs=[{"source_type": "transcript_or_document", "raw_text": profile_text}],
                )
                retrieved_background = retrieve_user_background(
                    user_id=user_id,
                    query=task_query,
                    recommended_chunk_types=[
                        "knowledge_boundary",
                        "expression_preference",
                        "technical_exposure",
                    ],
                )
                if retrieved_background and retrieved_background.get("structured_profile"):
                    inferred_role = retrieved_background["structured_profile"].get("role_lens", manual_role)
                    if allow_manual_override and manual_role != "general":
                        profile["role"] = manual_role
                    else:
                        profile["role"] = inferred_role
            except Exception as e:
                st.warning(f"Profile inference skipped: {e}")

        st.session_state.users[user_id]["profile"] = profile

        if use_rag_context or run_rag_comparison:
            rag = load_rag_if_needed()

        all_outputs = []
        all_outputs_no_rag = []
        generated_files = []

        progress = st.progress(0)
        total_sources = len(source_records)

        for s_idx, src in enumerate(source_records, start=1):
            participant_id = safe_filename(src["participant_id"])
            source_name = src["source_name"]
            source_kind = src["source_kind"]
            source_date = src["source_date"]
            text = src["source_text"]

            if not text.strip():
                st.warning(f"No text extracted from {source_name}.")
                continue

            human_aligned_mode = (
                use_human_segments_for_alignment
                and human_df_for_alignment is not None
                and not human_df_for_alignment.empty
            )

            if human_aligned_mode:
                chunks = build_human_aligned_chunks(
                    human_df_for_alignment,
                    participant_id=src.get("participant_id", participant_id),
                )
                if chunks:
                    st.info(
                        f"Using human CSV segmentation for alignment: {len(chunks)} segments. "
                        "This is recommended for valid F1/Kappa comparison."
                    )
                else:
                    st.warning(
                        "Human CSV was uploaded, but no usable text segments were found. "
                        "Falling back to automatic chunking."
                    )
                    human_aligned_mode = False
                    chunks = chunk_text(text)
            else:
                chunks = chunk_text(text)

            def run_single_coding_setting(use_rag_flag, setting_name):
                setting_output = []

                chunk_batches = batch_list(chunks, batch_size=BATCH_SIZE)

                for b_idx, chunk_batch in enumerate(chunk_batches, start=1):
                    source_section = (
                        f"{participant_id}|{source_kind}|"
                        f"{safe_filename(source_name)[:60]}|batch_{b_idx}|{setting_name}"
                    )

                    joined_batch_text = "\n\n".join(chunk_batch)

                    # Contextual RAG window:
                    # - Coding output remains restricted to the current batch.
                    # - Retrieval query includes previous/current/next batches to improve
                    #   local-context grounding for ambiguous qualitative segments.
                    retrieval_query = build_contextual_retrieval_query(
                        chunk_batches=chunk_batches,
                        batch_index=b_idx - 1,
                        window_size=1,
                        max_chars=1800,
                    )

                    retrieved_context = ""
                    if use_rag_flag and rag is not None:
                        retrieved_context = get_rag_context(
                            rag=rag,
                            query=retrieval_query,
                            profile=profile,
                            use_rag=True,
                        )

                    if show_context and use_rag_flag and b_idx == 1:
                        with st.expander(f"RAG context sample: {source_name}"):
                            st.text(retrieved_context[:3000])

                    try:
                        initial_segments = run_initial_llm_batch_coding(
                            segment_batch=chunk_batch,
                            codebook=CODEBOOK,
                            client=client,
                            source_section=source_section,
                            source_kind=source_kind,
                        )

                        if use_rag_flag:
                            coded_segments = run_rag_batch_refinement_coding(
                                initial_segments=initial_segments,
                                segment_batch=chunk_batch,
                                codebook=CODEBOOK,
                                client=client,
                                retrieved_context=retrieved_context,
                                source_section=source_section,
                                source_kind=source_kind,
                            )
                        else:
                            coded_segments = initial_segments

                        setting_output.extend(coded_segments)
                    except Exception as e:
                        st.error(
                            f"Batch LLM coding failed for {source_name}, batch {b_idx}, "
                            f"setting={setting_name}: {e}"
                        )
                        for chunk in chunk_batch:
                            setting_output.append({
                                "text": chunk,
                                "codes": ["ERROR"],
                                "rationale": str(e),
                                "source_section": source_section,
                                "source_type": source_kind,
                            })

                return setting_output

            if run_rag_comparison:
                aggregated_output_no_rag = run_single_coding_setting(False, "no_rag")
                aggregated_output = run_single_coding_setting(True, "rag")
            else:
                setting_name = "rag" if use_rag_context else "no_rag"
                aggregated_output = run_single_coding_setting(use_rag_context, setting_name)
                aggregated_output_no_rag = None

            df = make_coding_dataframe(
                aggregated_output,
                participant_id=participant_id,
                source_name=source_name,
                source_kind=source_kind,
                source_date=source_date,
            )

            df_no_rag = None
            if run_rag_comparison and aggregated_output_no_rag is not None:
                df_no_rag = make_coding_dataframe(
                    aggregated_output_no_rag,
                    participant_id=participant_id,
                    source_name=source_name,
                    source_kind=source_kind,
                    source_date=source_date,
                )

            base = safe_filename(f"{participant_id}_{source_kind}_{os.path.basename(str(source_name))[:50]}")
            json_file = os.path.join(user_output_dir, f"{base}_coding.json")
            csv_file = os.path.join(user_output_dir, f"{base}_coding.csv")
            no_rag_json_file = os.path.join(user_output_dir, f"{base}_no_rag_coding.json")
            no_rag_csv_file = os.path.join(user_output_dir, f"{base}_no_rag_coding.csv")

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(aggregated_output, f, indent=2, ensure_ascii=False)

            df.to_csv(csv_file, index=False)
            generated_files.extend([json_file, csv_file])
            all_outputs.append(df)

            if run_rag_comparison and df_no_rag is not None:
                with open(no_rag_json_file, "w", encoding="utf-8") as f:
                    json.dump(aggregated_output_no_rag, f, indent=2, ensure_ascii=False)

                df_no_rag.to_csv(no_rag_csv_file, index=False)
                generated_files.extend([no_rag_json_file, no_rag_csv_file])
                all_outputs_no_rag.append(df_no_rag)

                st.subheader(f"No-RAG Coding Output: {participant_id} / {source_kind}")
                st.dataframe(
                    df_no_rag[["segment_index", "source_type", "source_date", "text", "codes", "rationale"]].head(30)
                )

            st.subheader(f"Coding Output: {participant_id} / {source_kind}")
            st.dataframe(df[["segment_index", "source_type", "source_date", "text", "codes", "rationale"]].head(30))

            with open(csv_file, "rb") as f:
                st.download_button(
                    label=f"Download {base} CSV",
                    data=f,
                    file_name=f"{base}_coding.csv",
                    mime="text/csv",
                )

            progress.progress(s_idx / total_sources)

        if all_outputs:
            summary_df = pd.concat(all_outputs, ignore_index=True)
            summary_csv = os.path.join(user_output_dir, "batch_summary_coding.csv")
            summary_df.to_csv(summary_csv, index=False)
            generated_files.append(summary_csv)

            summary_no_rag_df = None
            if run_rag_comparison and all_outputs_no_rag:
                summary_no_rag_df = pd.concat(all_outputs_no_rag, ignore_index=True)
                summary_no_rag_csv = os.path.join(user_output_dir, "batch_summary_no_rag_coding.csv")
                summary_no_rag_df.to_csv(summary_no_rag_csv, index=False)
                generated_files.append(summary_no_rag_csv)

            code_counts = code_frequency(summary_df)
            group_counts = code_frequency_by_group(summary_df, "source_type")
            participant_counts = code_frequency_by_group(summary_df, "participant_id")

            st.subheader("Overall Code Frequency Table")
            st.dataframe(code_counts.reset_index().rename(columns={"index": "code", 0: "count"}))

            st.subheader("Overall Code Frequency Heatmap")
            if not code_counts.empty:
                fig, ax = plt.subplots(figsize=(10, 2))
                sns.heatmap(code_counts.to_frame().T, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
            else:
                st.info("No non-empty codes found for overall heatmap.")

            if not group_counts.empty:
                st.subheader("Grouped Heatmap: Interview vs Costing vs Policy")
                fig2, ax2 = plt.subplots(figsize=(10, max(2, 0.8 * len(group_counts))))
                sns.heatmap(group_counts, annot=True, fmt="d", cmap="Blues", ax=ax2)
                st.pyplot(fig2)

            if not participant_counts.empty:
                st.subheader("Participant-Level Code Heatmap")
                fig3, ax3 = plt.subplots(figsize=(10, max(2, 0.6 * len(participant_counts))))
                sns.heatmap(participant_counts, annot=True, fmt="d", cmap="Blues", ax=ax3)
                st.pyplot(fig3)

            if "source_date" in summary_df.columns and summary_df["source_date"].astype(str).str.len().gt(0).any():
                st.subheader("Time / Date-Level Code Distribution")
                date_counts = code_frequency_by_group(summary_df[summary_df["source_date"] != ""], "source_date")
                if not date_counts.empty:
                    st.dataframe(date_counts)
                    fig4, ax4 = plt.subplots(figsize=(10, max(2, 0.6 * len(date_counts))))
                    sns.heatmap(date_counts, annot=True, fmt="d", cmap="Blues", ax=ax4)
                    st.pyplot(fig4)

            topic_df = pd.DataFrame()
            doc_topic_df = pd.DataFrame()

            if run_lda:
                st.subheader("LDA Topic Modeling")
                topic_df, doc_topic_df = run_lda_topic_modeling(summary_df["text"].tolist(), n_topics=n_topics)
                if not topic_df.empty:
                    st.dataframe(topic_df)

                    topic_csv = os.path.join(user_output_dir, "lda_topics.csv")
                    topic_df.to_csv(topic_csv, index=False)
                    generated_files.append(topic_csv)

                    doc_topic_csv = os.path.join(user_output_dir, "lda_document_topics.csv")
                    doc_topic_df.to_csv(doc_topic_csv, index=False)
                    generated_files.append(doc_topic_csv)
                else:
                    st.info("Not enough segments for LDA topic modeling.")

            if run_bertopic:
                st.subheader("BERTopic")
                bertopic_info, bertopic_error = run_bertopic_optional(summary_df["text"].tolist())
                if bertopic_error:
                    st.warning(bertopic_error)
                else:
                    st.dataframe(bertopic_info)
                    bertopic_csv = os.path.join(user_output_dir, "bertopic_topics.csv")
                    bertopic_info.to_csv(bertopic_csv, index=False)
                    generated_files.append(bertopic_csv)

            comparison_df = None
            if human_coding_file is not None:
                try:
                    if human_df_for_alignment is not None:
                        human_df = human_df_for_alignment.copy()
                    else:
                        human_coding_file.seek(0)
                        human_df = pd.read_csv(human_coding_file)
                    comparison_df = compare_llm_human(summary_df, human_df, CODEBOOK)

                    comparison_csv = os.path.join(user_output_dir, "llm_vs_human_comparison.csv")
                    comparison_df.to_csv(comparison_csv, index=False)
                    generated_files.append(comparison_csv)

                    st.subheader("LLM vs Human Coding Comparison")
                    st.dataframe(comparison_df)

                    if run_rag_comparison and summary_no_rag_df is not None:
                        no_rag_comparison_df = compare_llm_human(summary_no_rag_df, human_df, CODEBOOK)

                        no_rag_comparison_csv = os.path.join(
                            user_output_dir, "no_rag_vs_human_comparison.csv"
                        )
                        no_rag_comparison_df.to_csv(no_rag_comparison_csv, index=False)
                        generated_files.append(no_rag_comparison_csv)

                        rag_vs_no_rag_table = build_rag_comparison_table(
                            no_rag_comparison=no_rag_comparison_df,
                            rag_comparison=comparison_df,
                        )

                        rag_vs_no_rag_csv = os.path.join(
                            user_output_dir, "rag_vs_no_rag_macro_comparison.csv"
                        )
                        rag_vs_no_rag_table.to_csv(rag_vs_no_rag_csv, index=False)
                        generated_files.append(rag_vs_no_rag_csv)

                        st.subheader("RAG vs No-RAG Macro-Level Comparison")
                        st.dataframe(rag_vs_no_rag_table)

                        try:
                            delta_f1 = rag_vs_no_rag_table.loc[1, "delta_macro_f1_vs_no_rag"]
                            delta_kappa = rag_vs_no_rag_table.loc[1, "delta_macro_kappa_vs_no_rag"]
                            st.info(
                                f"Compared with No-RAG, RAG changes Macro F1 by {delta_f1:.4f} "
                                f"and Macro Cohen's Kappa by {delta_kappa:.4f}."
                            )
                        except Exception:
                            st.info(
                                "RAG vs No-RAG comparison table generated. "
                                "Delta metrics could not be summarized automatically."
                            )

                    metric_df = comparison_df[
                        comparison_df["code"] != "MACRO_AVERAGE"
                    ][["code", "cohen_kappa", "f1"]].set_index("code")

                    if not metric_df.empty:
                        st.subheader("LLM vs Human Agreement Heatmap")
                        fig_agree, ax_agree = plt.subplots(figsize=(8, 3))
                        sns.heatmap(metric_df, annot=True, cmap="Blues", vmin=0, vmax=1, ax=ax_agree)
                        st.pyplot(fig_agree)

                    with open(comparison_csv, "rb") as f:
                        st.download_button(
                            "Download LLM vs Human Comparison CSV",
                            data=f,
                            file_name="llm_vs_human_comparison.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Human coding comparison failed: {e}")

            pdf_file = os.path.join(user_output_dir, "envio_coding_report.pdf")
            try:
                generate_pdf_report(summary_df, code_counts, group_counts, topic_df, comparison_df, pdf_file)
                generated_files.append(pdf_file)

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        data=f,
                        file_name="envio_coding_report.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.error(f"PDF report generation failed: {e}")

            zip_buffer = create_zip_from_files(generated_files)
            st.download_button(
                "Download All Outputs ZIP",
                data=zip_buffer,
                file_name=f"{safe_filename(user_id)}_envio_outputs.zip",
                mime="application/zip",
            )

            st.session_state.users[user_id]["outputs"].append(summary_df)

    else:
        try:
            if not query.strip():
                st.warning("Please enter a question for QA or Summary mode.")
                st.stop()

            rag = load_rag_if_needed()

            result = rag.answer_question(
                query=query,
                mode=mode,
                role=manual_role,
                user_profile=profile,
            )

            st.subheader("Final Answer")
            st.write(result.get("answer", "No answer returned."))

            if "base_explanation" in result:
                st.subheader("Base Explanation Before Expression Layer")
                st.write(result.get("base_explanation", ""))

            if "expression_plan" in result:
                st.subheader("Expression Layer Plan")
                st.json(result.get("expression_plan", {}))

            st.subheader("Top Citations")
            st.write(result.get("citations", []))

            if show_context:
                st.subheader("Retrieved Context")
                st.text(result.get("retrieved_context", ""))

        except Exception as e:
            st.error(f"{mode.upper()} failed: {e}")

    if show_debug:
        st.subheader("Session / Active Profile")
        st.json({
            "user_id": user_id,
            "output_dir": user_output_dir,
            "profile": profile,
            "merge_related_files": merge_related_files,
            "use_rag_context": use_rag_context,
            "run_rag_comparison": run_rag_comparison,
            "use_human_segments_for_alignment": use_human_segments_for_alignment,
            "uploaded_file_count": len(uploaded_files or []),
            "has_url_input": bool(url_input.strip()),
            "lda_enabled": run_lda,
            "bertopic_enabled": run_bertopic,
            "batch_size": BATCH_SIZE,
            "rag_context_char_limit": RAG_CONTEXT_CHAR_LIMIT,
            "contextual_rag_window_size": 1,
        })
