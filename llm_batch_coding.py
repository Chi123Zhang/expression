import os
import glob
import json
import re
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, f1_score

from rag_system import load_pdf, load_docx, initialize_rag

# =========================================================
# Paths
# =========================================================
transcript_folder = "./transcripts"       # LLM input PDF/DOCX/TXT files
human_folder = "./human_coding"           # Human coding JSON files
llm_output_folder = "./coding_outputs"    # LLM output JSON/CSV files
summary_csv = "./llm_vs_human_summary.csv"
os.makedirs(llm_output_folder, exist_ok=True)

# =========================================================
# Settings
# =========================================================
codebook = [
    "environmental_barrier",
    "social_support",
    "healthcare_access",
    "stigma",
    "mental_health",
]

CHUNK_CHARS = 3000
CHUNK_OVERLAP = 300

# Main switches
USE_RAG_REFINEMENT = True
RUN_PAIRED_RAG_COMPARISON = True

# Batch setting: 5 is the safest balance between speed and JSON stability.
BATCH_SIZE = 5

# Reduce retrieved context to lower token cost and timeout risk.
MAX_RAG_CONTEXT_CHARS = 2000

RAG_DOCS_DIR = "."

# Optional: if you have human CSV files and want exact human-aligned segment evaluation,
# put them in human_folder with names like: <base_name>_human.csv
# Required columns: segment_index, text, and code columns.
USE_HUMAN_CSV_AS_SEGMENT_TEMPLATE = True


# =========================================================
# OpenAI client
# =========================================================
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY not set.")
client = OpenAI(api_key=api_key)


# =========================================================
# Optional RAG initialization
# =========================================================
rag = None
if USE_RAG_REFINEMENT or RUN_PAIRED_RAG_COMPARISON:
    try:
        rag = initialize_rag(docs_dir=RAG_DOCS_DIR, force_rebuild=False)
    except Exception as e:
        print(f"[Warning] RAG initialization failed. Running No-RAG only. Reason: {e}")
        rag = None
        USE_RAG_REFINEMENT = False
        RUN_PAIRED_RAG_COMPARISON = False


# =========================================================
# File loading
# =========================================================
def load_transcript(file_path: str) -> str:
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        pages = load_pdf(file_path)
        return "\n".join(page_text for _, page_text in pages)
    if suffix == ".docx":
        pages = load_docx(file_path)
        return "\n".join(page_text for _, page_text in pages)
    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap_chars: int = CHUNK_OVERLAP) -> List[str]:
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


def batch_list(items: List[Any], batch_size: int = BATCH_SIZE) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# =========================================================
# JSON helpers
# =========================================================
def extract_json_object(content: str) -> Dict:
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


def clean_code_list(codes) -> List[str]:
    if isinstance(codes, list):
        raw = codes
    else:
        raw = str(codes).replace(";", ",").split(",")

    allowed = set(codebook)
    return [str(c).strip() for c in raw if str(c).strip() in allowed]


def call_openai_json(
    prompt: str,
    system: str = "You are a careful qualitative coding assistant. Return strict JSON only.",
) -> Dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return extract_json_object(response.choices[0].message.content.strip())


def normalize_segments(
    parsed,
    fallback_text: str,
    source_section: str = "",
    source_type: str = "interview",
    expected_texts: List[str] | None = None,
) -> List[Dict]:
    """
    Normalize model JSON into a list of segment dictionaries.

    If expected_texts is provided, the function preserves one output row per expected input
    segment. This prevents segment drift during batch coding.
    """
    if isinstance(parsed, list):
        segments = parsed
    elif isinstance(parsed, dict):
        segments = parsed.get("segments", [])
    else:
        segments = []

    output = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        output.append({
            "text": str(seg.get("text", "")).strip(),
            "codes": clean_code_list(seg.get("codes", [])),
            "rationale": str(seg.get("rationale", "")).strip(),
            "source_section": seg.get("source_section", source_section),
            "source_type": seg.get("source_type", source_type),
        })

    # For batch coding, force exact length and preserve original segment text.
    if expected_texts is not None:
        fixed = []
        for i, original_text in enumerate(expected_texts):
            if i < len(output):
                row = output[i]
                fixed.append({
                    "text": original_text,
                    "codes": clean_code_list(row.get("codes", [])),
                    "rationale": row.get("rationale", ""),
                    "source_section": source_section,
                    "source_type": source_type,
                })
            else:
                fixed.append({
                    "text": original_text,
                    "codes": [],
                    "rationale": "No coding result was returned for this segment in the batch.",
                    "source_section": source_section,
                    "source_type": source_type,
                })
        return fixed

    if not output:
        return [{
            "text": fallback_text,
            "codes": [],
            "rationale": "No codable segment was returned.",
            "source_section": source_section,
            "source_type": source_type,
        }]

    return output


# =========================================================
# Human coding loading
# =========================================================
def find_human_file(base_name: str) -> str | None:
    csv_path = os.path.join(human_folder, f"{base_name}_human.csv")
    json_path = os.path.join(human_folder, f"{base_name}_human.json")

    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(json_path):
        return json_path
    return None


def load_human_csv_segments(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        raise ValueError(f"Human CSV must contain a 'text' column: {path}")

    if "segment_index" in df.columns:
        df = df.sort_values("segment_index")

    return df["text"].fillna("").astype(str).tolist()


def load_human_codes(path: str) -> List[set]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        if "segment_index" in df.columns:
            df = df.sort_values("segment_index")

        if "codes" in df.columns:
            return [set(clean_code_list(x)) for x in df["codes"].fillna("")]

        rows = []
        for _, row in df.iterrows():
            codes = set()
            for c in codebook:
                if c in df.columns:
                    try:
                        if int(row[c]) == 1:
                            codes.add(c)
                    except Exception:
                        pass
            rows.append(codes)
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [set(clean_code_list(item.get("codes", []))) for item in data]


def load_llm_codes_json(path: str) -> List[set]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [set(clean_code_list(item.get("codes", []))) for item in data]


# =========================================================
# RAG retrieval
# =========================================================
def get_rag_context(text_batch: str) -> str:
    if rag is None:
        return ""

    try:
        result = rag.answer_question(
            query=text_batch[:1200],
            mode="coding",
            role="general",
            user_profile={
                "role": "general",
                "technical_level": "medium",
                "goal": "coding",
            },
        )
        return result.get("retrieved_context", "")
    except Exception as e:
        print(f"[Warning] RAG retrieval skipped for one batch: {e}")
        return ""


# =========================================================
# Stage 1: No-RAG batch coding
# =========================================================
def run_initial_llm_batch_coding(
    segment_batch: List[str],
    source_section: str = "",
    source_type: str = "interview",
) -> List[Dict]:
    """
    Batch qualitative coding.

    Multiple aligned segments are coded in a single API call.
    The function preserves one coding result per input segment.
    """
    formatted_segments = []
    for i, seg in enumerate(segment_batch, start=1):
        formatted_segments.append(f"SEGMENT {i}:\n{seg}\n")

    joined_segments = "\n\n".join(formatted_segments)

    prompt = f"""
You are an expert qualitative research coding assistant for the ENVIO / TechMPower study.

Research goal:
Compare LLM-generated qualitative coding against human coding for de-identified qualitative transcripts.
The focus is how environmental, institutional, social, mental health, stigma, and healthcare factors influence HIV care engagement.

Allowed codebook:
{codebook}

Task:
Code each segment independently.

Important rules:
1. Preserve segment boundaries.
2. Return exactly one coding result per input segment and keep the same order.
3. Use ONLY the allowed codebook.
4. Multi-label coding is allowed.
5. Do not invent codes outside the codebook.
6. Do not merge segments.
7. Do not split a segment into multiple output rows.
8. Output ONLY valid JSON. No markdown. No explanation outside JSON.

Segments:
{joined_segments}

Required JSON schema:
{{
  "segments": [
    {{
      "text": "original segment text",
      "codes": ["code1", "code2"],
      "rationale": "brief reason grounded in the transcript text",
      "source_section": "{source_section}",
      "source_type": "{source_type}"
    }}
  ]
}}
"""
    parsed = call_openai_json(prompt)
    return normalize_segments(
        parsed,
        fallback_text="BATCH_SEGMENT",
        source_section=source_section,
        source_type=source_type,
        expected_texts=segment_batch,
    )


# =========================================================
# Stage 2: RAG batch refinement
# =========================================================
def run_rag_refinement_batch_coding(
    initial_segments: List[Dict],
    segment_batch: List[str],
    retrieved_context: str,
    source_section: str = "",
    source_type: str = "interview",
) -> List[Dict]:
    """
    Batch RAG refinement.

    It reviews batch initial coding against RAG context while preserving one row per input segment.
    """
    if not retrieved_context or not str(retrieved_context).strip():
        return initial_segments

    initial_json = json.dumps({"segments": initial_segments}, ensure_ascii=False, indent=2)
    joined_segments = "\n\n".join([f"SEGMENT {i}:\n{text}" for i, text in enumerate(segment_batch, start=1)])

    prompt = f"""
You are refining qualitative coding for the ENVIO / TechMPower study.

Important principle:
The transcript text is the primary evidence.
The RAG context is supporting project/codebook context.
Use RAG to clarify code definitions, study framing, and borderline cases.
Do not add codes that are not supported by the transcript segment.

Allowed codebook:
{codebook}

Original input segments:
{joined_segments}

Initial No-RAG coding result:
{initial_json}

Relevant RAG context from project documents:
{retrieved_context[:MAX_RAG_CONTEXT_CHARS]}

Refinement instructions:
1. Review each input segment and its initial coding.
2. Preserve segment boundaries.
3. Return exactly one coding result per input segment and keep the same order.
4. Keep the initial coding if it is already well supported.
5. Revise codes only when the RAG context clearly improves consistency with the codebook or study framing.
6. You may add, remove, or replace codes, but only using the allowed codebook.
7. Do not merge segments.
8. Do not split a segment into multiple output rows.
9. Output ONLY valid JSON. No markdown. No explanation outside JSON.

Required JSON schema:
{{
  "segments": [
    {{
      "text": "original segment text",
      "codes": ["code1", "code2"],
      "rationale": "kept/refined reason grounded in transcript + RAG context",
      "source_section": "{source_section}",
      "source_type": "{source_type}"
    }}
  ]
}}
"""
    parsed = call_openai_json(
        prompt,
        system="You are a careful qualitative coding refinement assistant. Return strict JSON only.",
    )
    return normalize_segments(
        parsed,
        fallback_text="BATCH_SEGMENT",
        source_section=source_section,
        source_type=source_type,
        expected_texts=segment_batch,
    )


def run_two_stage_batch_coding(
    segment_batch: List[str],
    use_rag: bool = True,
    source_section: str = "",
    source_type: str = "interview",
) -> List[Dict]:
    initial_segments = run_initial_llm_batch_coding(
        segment_batch=segment_batch,
        source_section=source_section,
        source_type=source_type,
    )

    if not use_rag:
        return initial_segments

    joined_batch_text = "\n\n".join(segment_batch)
    retrieved_context = get_rag_context(joined_batch_text)

    return run_rag_refinement_batch_coding(
        initial_segments=initial_segments,
        segment_batch=segment_batch,
        retrieved_context=retrieved_context,
        source_section=source_section,
        source_type=source_type,
    )


def save_segments_json(path: str, segments: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)


# =========================================================
# Run LLM coding for all transcripts
# =========================================================
for transcript_file in glob.glob(os.path.join(transcript_folder, "*.*")):
    base_name = os.path.splitext(os.path.basename(transcript_file))[0]
    print(f"\nProcessing {base_name} ...")

    human_file = find_human_file(base_name)

    if USE_HUMAN_CSV_AS_SEGMENT_TEMPLATE and human_file and human_file.lower().endswith(".csv"):
        print(f"Using human CSV segmentation template: {human_file}")
        chunks = load_human_csv_segments(human_file)
    else:
        text = load_transcript(transcript_file)
        chunks = chunk_text(text)

    if not chunks:
        print(f"[Warning] No text segments found for {base_name}. Skipping.")
        continue

    batches = batch_list(chunks, BATCH_SIZE)
    print(f"Total segments: {len(chunks)} | Batch size: {BATCH_SIZE} | Total batches: {len(batches)}")

    aggregated_rag = []
    aggregated_no_rag = []

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"  Coding batch {batch_idx}/{len(batches)} ...")

        section_no_rag = f"{base_name}|batch_{batch_idx}|no_rag"
        no_rag_segments = run_two_stage_batch_coding(
            batch,
            use_rag=False,
            source_section=section_no_rag,
        )
        aggregated_no_rag.extend(no_rag_segments)

        if USE_RAG_REFINEMENT:
            section_rag = f"{base_name}|batch_{batch_idx}|rag_refinement"
            rag_segments = run_two_stage_batch_coding(
                batch,
                use_rag=True,
                source_section=section_rag,
            )
            aggregated_rag.extend(rag_segments)
        else:
            aggregated_rag.extend(no_rag_segments)

    rag_output_path = os.path.join(llm_output_folder, f"{base_name}_coding.json")
    no_rag_output_path = os.path.join(llm_output_folder, f"{base_name}_no_rag_coding.json")

    save_segments_json(rag_output_path, aggregated_rag)
    save_segments_json(no_rag_output_path, aggregated_no_rag)

    rag_csv_path = os.path.join(llm_output_folder, f"{base_name}_coding.csv")
    no_rag_csv_path = os.path.join(llm_output_folder, f"{base_name}_no_rag_coding.csv")

    pd.DataFrame(aggregated_rag).to_csv(rag_csv_path, index=False)
    pd.DataFrame(aggregated_no_rag).to_csv(no_rag_csv_path, index=False)

    print(f"Saved RAG/refined coding JSON: {rag_output_path}")
    print(f"Saved No-RAG coding JSON: {no_rag_output_path}")
    print(f"Saved RAG/refined coding CSV: {rag_csv_path}")
    print(f"Saved No-RAG coding CSV: {no_rag_csv_path}")


# =========================================================
# Compute LLM vs Human consistency
# =========================================================
def compare_code_sets(llm_codes, human_codes, transcript_name, setting):
    rows = []
    min_len = min(len(llm_codes), len(human_codes))

    llm_codes = llm_codes[:min_len]
    human_codes = human_codes[:min_len]

    for code in codebook:
        llm_binary = [1 if code in s else 0 for s in llm_codes]
        human_binary = [1 if code in s else 0 for s in human_codes]

        try:
            kappa = cohen_kappa_score(human_binary, llm_binary)
        except Exception:
            kappa = None

        try:
            f1 = f1_score(human_binary, llm_binary, zero_division=0)
        except Exception:
            f1 = None

        rows.append({
            "transcript": transcript_name,
            "setting": setting,
            "code": code,
            "cohen_kappa": kappa,
            "f1_score": f1,
            "human_positive": sum(human_binary),
            "llm_positive": sum(llm_binary),
            "matched_segments": min_len,
        })

    return rows


summary_list = []
for rag_file in glob.glob(os.path.join(llm_output_folder, "*_coding.json")):
    base_name = os.path.basename(rag_file).replace("_coding.json", "")
    if base_name.endswith("_no_rag"):
        continue

    human_file = find_human_file(base_name)
    if not human_file:
        print(f"Skipping {base_name}, human coding not found.")
        continue

    human_codes = load_human_codes(human_file)

    rag_codes = load_llm_codes_json(rag_file)
    summary_list.extend(compare_code_sets(rag_codes, human_codes, base_name, "RAG"))

    no_rag_file = os.path.join(llm_output_folder, f"{base_name}_no_rag_coding.json")
    if os.path.exists(no_rag_file):
        no_rag_codes = load_llm_codes_json(no_rag_file)
        summary_list.extend(compare_code_sets(no_rag_codes, human_codes, base_name, "No RAG"))


# =========================================================
# Save summary CSV + macro comparison
# =========================================================
df = pd.DataFrame(summary_list)
df.to_csv(summary_csv, index=False)
print(f"\nSaved summary CSV: {summary_csv}")

if not df.empty:
    macro_df = (
        df.groupby("setting")[["cohen_kappa", "f1_score"]]
        .mean()
        .reset_index()
        .rename(columns={"cohen_kappa": "macro_kappa", "f1_score": "macro_f1"})
    )

    macro_csv = "./rag_vs_no_rag_macro_summary.csv"
    macro_df.to_csv(macro_csv, index=False)

    print("\nMacro comparison:")
    print(macro_df)

    for setting in sorted(df["setting"].unique()):
        setting_df = df[df["setting"] == setting]

        kappa_pivot = setting_df.pivot(index="transcript", columns="code", values="cohen_kappa")
        plt.figure(figsize=(10, 6))
        sns.heatmap(kappa_pivot, annot=True, cmap="coolwarm", center=0, cbar_kws={"label": "Cohen's kappa"})
        plt.title(f"{setting}: LLM vs Human Coding Cohen's Kappa")
        plt.tight_layout()
        plt.savefig(
            os.path.join(llm_output_folder, f"{setting.replace(' ', '_').lower()}_kappa_heatmap.png"),
            dpi=200,
        )
        plt.close()

        f1_pivot = setting_df.pivot(index="transcript", columns="code", values="f1_score")
        plt.figure(figsize=(10, 6))
        sns.heatmap(f1_pivot, annot=True, cmap="YlGnBu", vmin=0, vmax=1, cbar_kws={"label": "F1 Score"})
        plt.title(f"{setting}: LLM vs Human Coding F1 Score")
        plt.tight_layout()
        plt.savefig(
            os.path.join(llm_output_folder, f"{setting.replace(' ', '_').lower()}_f1_heatmap.png"),
            dpi=200,
        )
        plt.close()

    avg_metrics = (
        df.groupby(["setting", "code"])
        .agg({"cohen_kappa": "mean", "f1_score": "mean"})
        .reset_index()
    )

    avg_csv = "./rag_vs_no_rag_code_level_summary.csv"
    avg_metrics.to_csv(avg_csv, index=False)

    print("\nAverage consistency per code:")
    print(avg_metrics)
    print(f"\nSaved macro summary: {macro_csv}")
    print(f"Saved code-level summary: {avg_csv}")
else:
    print("No human comparison rows were generated. Check human_coding folder and filename format.")
