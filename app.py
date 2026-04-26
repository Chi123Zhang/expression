# app.py
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
from fpdf import FPDF
from sklearn.metrics import cohen_kappa_score, f1_score

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background


st.set_page_config(page_title="ENVIO LLM Coding Assistant", layout="wide")
st.title("ENVIO LLM Coding Assistant")
st.caption("Document-grounded LLM coding and topic modeling for HIV care engagement transcripts")


# -----------------------------
# Utilities
# -----------------------------
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


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


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


def run_llm_coding_with_context(text_segment, codebook, client, retrieved_context="", source_section=""):
    prompt = f"""
You are an expert qualitative research coding assistant for the ENVIO / TechMPower study.

Research goal:
Compare LLM-generated qualitative coding against human coding for de-identified interview transcripts.
The substantive focus is how environmental, institutional, social, and healthcare factors influence HIV care engagement and related implementation processes.

Allowed codebook:
{codebook}

Transcript source section:
{source_section}

Transcript segment:
{text_segment}

Relevant RAG context from project documents:
{retrieved_context}

Instructions:
1. Split the transcript segment into meaningful qualitative units.
2. Assign zero, one, or multiple codes from the allowed codebook only.
3. Do not invent codes outside the codebook.
4. Preserve short evidence excerpts.
5. Output ONLY valid JSON. No markdown. No explanation.

Required JSON schema:
{{
  "segments": [
    {{
      "text": "meaningful transcript unit",
      "codes": ["code1", "code2"],
      "rationale": "brief reason grounded in the text",
      "source_section": "{source_section}"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a careful qualitative coding assistant. Return strict JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        parsed = extract_json_object(content)
        return parsed.get("segments", [])
    except Exception:
        return [
            {
                "text": text_segment,
                "codes": ["PARSE_ERROR"],
                "rationale": "The model output could not be parsed as JSON.",
                "source_section": source_section,
                "raw": content,
            }
        ]


def build_combined_transcript(uploaded_files, merge_files=True):
    if not uploaded_files:
        return []

    if not merge_files:
        records = []
        for f in uploaded_files:
            text = load_transcript_text(f)
            records.append(
                {
                    "participant_id": os.path.splitext(f.name)[0],
                    "file_name": f.name,
                    "text": f"\n\n[SECTION: {f.name}]\n{text}",
                }
            )
        return records

    participant_id = infer_participant_id([f.name for f in uploaded_files])
    combined_text = ""
    file_names = []

    for f in uploaded_files:
        text = load_transcript_text(f)
        file_names.append(f.name)
        combined_text += f"\n\n[SECTION: {f.name}]\n{text}\n"

    return [
        {
            "participant_id": participant_id,
            "file_name": " + ".join(file_names),
            "text": combined_text,
        }
    ]


def infer_participant_id(file_names):
    joined = " ".join(file_names)
    match = re.search(r"\b(\d{3}[A-Za-z]?)\b", joined)
    if match:
        return match.group(1)
    base = os.path.splitext(file_names[0])[0]
    return safe_filename(base[:40])


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


def make_coding_dataframe(segments, participant_id, file_name):
    rows = []
    for i, seg in enumerate(segments, start=1):
        codes = seg.get("codes", [])
        if not isinstance(codes, list):
            codes = [str(codes)]

        rows.append(
            {
                "participant_id": participant_id,
                "segment_index": i,
                "file_name": file_name,
                "source_section": seg.get("source_section", ""),
                "text": seg.get("text", ""),
                "codes": ",".join(codes),
                "rationale": seg.get("rationale", ""),
            }
        )
    return pd.DataFrame(rows)


def code_frequency(df):
    if df.empty or "codes" not in df.columns:
        return pd.Series(dtype=int)
    return df["codes"].fillna("").str.get_dummies(sep=",").sum().sort_values(ascending=False)


def compare_llm_human(llm_df, human_df, codebook):
    results = []

    if "segment_index" in human_df.columns and "segment_index" in llm_df.columns:
        merged = pd.merge(
            llm_df,
            human_df,
            on="segment_index",
            suffixes=("_llm", "_human"),
            how="inner",
        )
        llm_col = "codes_llm"
        human_col = "codes_human"
    else:
        n = min(len(llm_df), len(human_df))
        merged = pd.DataFrame(
            {
                "codes_llm": llm_df["codes"].head(n).values,
                "codes_human": human_df["codes"].head(n).values,
            }
        )
        llm_col = "codes_llm"
        human_col = "codes_human"

    for code in codebook:
        y_llm = merged[llm_col].fillna("").apply(lambda x: int(code in str(x).split(",")))
        y_human = merged[human_col].fillna("").apply(lambda x: int(code in str(x).split(",")))

        try:
            kappa = cohen_kappa_score(y_human, y_llm)
        except Exception:
            kappa = None

        try:
            f1 = f1_score(y_human, y_llm, zero_division=0)
        except Exception:
            f1 = None

        results.append(
            {
                "code": code,
                "cohen_kappa": kappa,
                "f1": f1,
                "human_positive": int(y_human.sum()),
                "llm_positive": int(y_llm.sum()),
            }
        )

    return pd.DataFrame(results)


def generate_pdf_report(summary_df, code_counts, comparison_df=None, output_file="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, txt="ENVIO LLM Coding Summary Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt=f"Total coded segments: {len(summary_df)}", ln=True)
    pdf.cell(0, 8, txt=f"Total files/participants represented: {summary_df['participant_id'].nunique()}", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, txt="Code Frequencies:", ln=True)
    for code, count in code_counts.items():
        safe_line = f"{code}: {int(count)}"
        pdf.cell(0, 7, txt=safe_line[:110], ln=True)

    if comparison_df is not None and not comparison_df.empty:
        pdf.ln(5)
        pdf.cell(0, 8, txt="LLM vs Human Coding Comparison:", ln=True)
        for _, row in comparison_df.iterrows():
            line = (
                f"{row['code']} | Kappa={row['cohen_kappa']} | "
                f"F1={row['f1']} | Human+={row['human_positive']} | LLM+={row['llm_positive']}"
            )
            pdf.cell(0, 7, txt=line[:110], ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, txt="Sample Segments:", ln=True)
    for _, row in summary_df.head(8).iterrows():
        text = str(row["text"]).replace("\n", " ")[:500]
        codes = str(row["codes"])
        pdf.multi_cell(0, 6, txt=f"Codes: {codes}\nText: {text}")
        pdf.ln(2)

    pdf.output(output_file)
    return output_file


def create_zip_from_files(file_paths):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            if os.path.exists(path):
                zf.write(path, arcname=os.path.basename(path))
    buffer.seek(0)
    return buffer


# -----------------------------
# Init RAG
# -----------------------------
if "rag" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = initialize_rag(docs_dir=".", force_rebuild=False)

rag = st.session_state.rag


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

mode = st.sidebar.selectbox("Choose mode", ["coding", "qa", "summary"])
manual_role = st.sidebar.selectbox("Choose response perspective", ["general", "pm", "engineer", "business"])

uploaded_files = st.sidebar.file_uploader(
    "Upload transcript files (PDF/DOCX/TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

merge_related_files = st.sidebar.checkbox(
    "Merge uploaded files as one participant/interview",
    value=True,
)

use_rag_context = st.sidebar.checkbox("Use RAG context for coding", value=True)
show_context = st.sidebar.checkbox("Show retrieved context", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=True)

human_coding_file = st.sidebar.file_uploader(
    "Optional: upload human coding CSV for comparison",
    type=["csv"],
)

use_resume_profile = st.sidebar.checkbox("Use uploaded transcript to infer profile", value=True)
allow_manual_override = st.sidebar.checkbox("Allow manual role override", value=True)
user_id = st.sidebar.text_input("User ID", value="demo_user")

query = st.text_area("Enter your question or coding instruction (optional)", height=120)


# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
        st.stop()

    client = OpenAI(api_key=api_key)

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
        if not uploaded_files:
            st.warning("Please upload at least one transcript file.")
            st.stop()

        transcript_records = build_combined_transcript(
            uploaded_files,
            merge_files=merge_related_files,
        )

        all_outputs = []
        generated_files = []

        progress = st.progress(0)
        total_records = len(transcript_records)

        for r_idx, record in enumerate(transcript_records, start=1):
            participant_id = safe_filename(record["participant_id"])
            fname_base = safe_filename(record["file_name"])
            text = record["text"]

            if not text.strip():
                st.warning(f"No text extracted from {record['file_name']}.")
                continue

            if use_resume_profile:
                try:
                    onboard_user_background(
                        user_id=user_id,
                        raw_background_inputs=[{"source_type": "transcript", "raw_text": text[:8000]}],
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

            chunks = chunk_text(text)
            aggregated_output = []

            for c_idx, chunk in enumerate(chunks, start=1):
                retrieved_context = get_rag_context(
                    rag=rag,
                    query=chunk[:1200],
                    profile=profile,
                    use_rag=use_rag_context,
                )

                if show_context and c_idx == 1:
                    with st.expander("Retrieved RAG context sample"):
                        st.text(retrieved_context[:3000])

                try:
                    coded_segments = run_llm_coding_with_context(
                        text_segment=chunk,
                        codebook=CODEBOOK,
                        client=client,
                        retrieved_context=retrieved_context,
                        source_section=f"{participant_id}_chunk_{c_idx}",
                    )
                    aggregated_output.extend(coded_segments)
                except Exception as e:
                    st.error(f"LLM coding failed for {participant_id}, chunk {c_idx}: {e}")
                    aggregated_output.append(
                        {
                            "text": chunk,
                            "codes": ["ERROR"],
                            "rationale": str(e),
                            "source_section": f"{participant_id}_chunk_{c_idx}",
                        }
                    )

            df = make_coding_dataframe(
                aggregated_output,
                participant_id=participant_id,
                file_name=record["file_name"],
            )

            json_file = os.path.join(user_output_dir, f"{participant_id}_coding.json")
            csv_file = os.path.join(user_output_dir, f"{participant_id}_coding.csv")

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(aggregated_output, f, indent=2, ensure_ascii=False)

            df.to_csv(csv_file, index=False)

            generated_files.extend([json_file, csv_file])
            all_outputs.append(df)

            st.subheader(f"Coding Output for {participant_id}")
            st.dataframe(df[["segment_index", "source_section", "text", "codes", "rationale"]].head(30))

            with open(csv_file, "rb") as f:
                st.download_button(
                    label=f"Download {participant_id} CSV",
                    data=f,
                    file_name=f"{participant_id}_coding.csv",
                    mime="text/csv",
                )

            progress.progress(r_idx / total_records)

        if all_outputs:
            summary_df = pd.concat(all_outputs, ignore_index=True)
            summary_csv = os.path.join(user_output_dir, "batch_summary_coding.csv")
            summary_df.to_csv(summary_csv, index=False)
            generated_files.append(summary_csv)

            code_counts = code_frequency(summary_df)

            st.subheader("Batch Code Frequency Table")
            st.dataframe(code_counts.reset_index().rename(columns={"index": "code", 0: "count"}))

            st.subheader("Batch Code Frequency Heatmap")
            fig, ax = plt.subplots(figsize=(10, 2))
            sns.heatmap(code_counts.to_frame().T, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            comparison_df = None
            if human_coding_file is not None:
                try:
                    human_df = pd.read_csv(human_coding_file)
                    comparison_df = compare_llm_human(summary_df, human_df, CODEBOOK)
                    comparison_csv = os.path.join(user_output_dir, "llm_vs_human_comparison.csv")
                    comparison_df.to_csv(comparison_csv, index=False)
                    generated_files.append(comparison_csv)

                    st.subheader("LLM vs Human Coding Comparison")
                    st.dataframe(comparison_df)

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
                generate_pdf_report(summary_df, code_counts, comparison_df, pdf_file)
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

            result = rag.answer_question(
                query=query,
                mode=mode,
                role=manual_role,
                user_profile=profile,
            )

            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned."))

            st.subheader("Top Citations")
            st.write(result.get("citations", []))

            if show_context:
                st.subheader("Retrieved Context")
                st.text(result.get("retrieved_context", ""))

        except Exception as e:
            st.error(f"{mode.upper()} failed: {e}")

    if show_debug:
        st.subheader("Session / Active Profile")
        st.json(
            {
                "user_id": user_id,
                "output_dir": user_output_dir,
                "profile": profile,
                "merge_related_files": merge_related_files,
                "use_rag_context": use_rag_context,
                "uploaded_file_count": len(uploaded_files or []),
            }
        )
