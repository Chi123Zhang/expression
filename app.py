import os
import glob
import tempfile
import json
import streamlit as st
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background
from query_orchestrator import process_query

st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")
st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with background-aware personalization")

# -----------------------------
# Helpers
# -----------------------------
def load_transcript_text(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            pages = load_pdf(tmp_path)
            text = " ".join(page_text for _, page_text in pages)
        elif suffix == ".docx":
            pages = load_docx(tmp_path)
            text = " ".join(page_text for _, page_text in pages)
        elif suffix == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            return ""
        return text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_llm_coding(text_segment, codebook, client):
    prompt = f"""
You are a qualitative research coding assistant.

Transcript segment:
{text_segment}

Task:
- Assign thematic codes to each meaningful segment.
- Use the following codebook: {codebook}
- Output JSON format:
[
  {{"text": "Segment text here","codes":["relevant_code1","relevant_code2"]}}
]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        messages=[{"role":"system","content":"Expert qualitative coding assistant."},
                  {"role":"user","content":prompt}]
    )
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [{"text": text_segment, "codes": ["PARSE_ERROR"], "raw": content}]

def display_citations(citations):
    if citations:
        for c in citations:
            page = f"p.{c['page']}" if c.get("page") else "doc"
            st.write(f"{c.get('source_file','unknown')} | {page} | {c.get('section','unknown')} | {c.get('aim','unknown')} | score={c.get('score','n/a')}")
    else:
        st.write("No citations available.")

def build_user_profile_from_background(retrieved_background: dict) -> dict:
    structured = (retrieved_background or {}).get("structured_profile") or {}
    role = structured.get("role_lens", "general")
    if role == "product_manager":
        role = "pm"
    return {"role": role, "technical_level": structured.get("technical_depth","medium"), "goal":"understanding", "short_reason":structured.get("short_reason","")}

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
mode = st.sidebar.selectbox("Choose mode", ["qa", "summary", "coding"])
manual_role = st.sidebar.selectbox("Choose response perspective", ["general", "pm", "engineer", "business"])
show_context = st.sidebar.checkbox("Show retrieved context", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=True)
uploaded_file = st.sidebar.file_uploader("Upload transcript (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
batch_folder = st.sidebar.text_input("Batch folder path for transcripts (optional)")
use_resume_profile = st.sidebar.checkbox("Use uploaded resume to infer profile", value=True)
allow_manual_override = st.sidebar.checkbox("Allow manual role override", value=True)
user_id = st.sidebar.text_input("User ID", value="demo_user")
query = st.text_area("Enter your question or paste transcript for coding", height=140)

# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    # -----------------------------
    # Batch processing模式
    # -----------------------------
    if batch_folder and mode=="coding":
        st.info(f"Processing batch folder: {batch_folder}")
        output_folder = os.path.join(batch_folder, "llm_outputs")
        os.makedirs(output_folder, exist_ok=True)
        codebook = ["environmental_barrier","social_support","healthcare_access","stigma","mental_health"]
        summary_list = []

        for f in glob.glob(os.path.join(batch_folder,"*.*")):
            base_name = os.path.splitext(os.path.basename(f))[0]
            text = load_transcript_text(open(f,"rb"))
            chunks = [text[i:i+2000] for i in range(0,len(text),2000)]
            aggregated_output = []
            for chunk in chunks:
                aggregated_output.extend(run_llm_coding(chunk, codebook, client))
            output_file = os.path.join(output_folder,f"{base_name}_coding.json")
            with open(output_file,"w",encoding="utf-8") as of:
                json.dump(aggregated_output,of,indent=2,ensure_ascii=False)
            st.write(f"Saved LLM coding for {base_name}")

        st.success(f"Batch processing done. Outputs in {output_folder}")

    # -----------------------------
    # 单文件交互模式
    # -----------------------------
    elif query.strip() or uploaded_file:
        inferred_profile = None
        effective_role = manual_role
        retrieved_background = None

        # 背景记忆
        if uploaded_file and use_resume_profile:
            transcript_text = load_transcript_text(uploaded_file)
            onboard_user_background(user_id=user_id, raw_background_inputs=[{"source_type":"resume","raw_text":transcript_text}])

        orchestration_result = process_query(user_id=user_id, raw_query=query, has_uploaded_project_doc=True)
        routing_decision = orchestration_result["routing_decision"]

        # 决定角色
        if "background_request" in routing_decision:
            bg_req = routing_decision["background_request"]
            retrieved_background = retrieve_user_background(user_id=bg_req["user_id"], query=bg_req["query"], recommended_chunk_types=bg_req["recommended_background_chunk_types"])
            if retrieved_background.get("structured_profile"):
                inferred_profile = build_user_profile_from_background(retrieved_background)
        if inferred_profile:
            effective_role = manual_role if (allow_manual_override and manual_role!="general") else inferred_profile["role"]

        # 显示 debug
        if show_debug:
            st.subheader("Routing Decision / Active Profile")
            st.json({"routing": routing_decision, "profile": inferred_profile, "effective_role": effective_role})

        # 生成输出
        if mode=="coding":
            codebook = ["environmental_barrier","social_support","healthcare_access","stigma","mental_health"]
            text = transcript_text if uploaded_file else query
            chunks = [text[i:i+2000] for i in range(0,len(text),2000)]
            aggregated_output = []
            for chunk in chunks:
                aggregated_output.extend(run_llm_coding(chunk, codebook, client))
            st.subheader("Coding Output (JSON)")
            st.json(aggregated_output)
        else:
            # QA/Summary模式
            result = rag.answer_question(query=query, mode=mode, role=effective_role, user_profile=inferred_profile)
            st.subheader("Answer")
            st.write(result["answer"])
            st.subheader("Top Citations")
            display_citations(result.get("citations",[]))
            if show_context:
                st.subheader("Retrieved Context")
                st.text(result.get("retrieved_context",""))
