import os
import glob
import tempfile
import json
import streamlit as st
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background
from query_orchestrator import process_query

st.set_page_config(page_title="ENVIO LLM Coding Assistant", layout="wide")
st.title("ENVIO LLM Coding Assistant")
st.caption("Document-grounded LLM coding and topic modeling for HIV care engagement transcripts")

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

def run_llm_coding_with_context(text_segment, codebook, client, retrieved_context):
    prompt = f"""
You are a qualitative research coding assistant.

Transcript segment:
{text_segment}

Relevant context from project documents:
{retrieved_context}

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

def generate_batch_summary(output_folder):
    all_csv_files = glob.glob(os.path.join(output_folder,"*_coding.csv"))
    if not all_csv_files:
        return None, None
    all_dfs = [pd.read_csv(f) for f in all_csv_files]
    summary_df = pd.concat(all_dfs, ignore_index=True)
    code_counts = summary_df['codes'].str.get_dummies(sep=',').sum()
    return summary_df, code_counts

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
show_context = st.sidebar.checkbox("Show retrieved context", value=False)
show_debug = st.sidebar.checkbox("Show debug info", value=True)
uploaded_file = st.sidebar.file_uploader("Upload transcript (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
batch_folder = st.sidebar.text_input("Batch folder path for transcripts (optional)")
use_resume_profile = st.sidebar.checkbox("Use uploaded resume to infer profile", value=True)
allow_manual_override = st.sidebar.checkbox("Allow manual role override", value=True)
user_id = st.sidebar.text_input("User ID", value="demo_user")
query = st.text_area("Enter your question or paste transcript for coding (optional)")

# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    codebook = ["environmental_barrier","social_support","healthcare_access","stigma","mental_health"]

    # 默认 query
    if not query.strip() and uploaded_file and mode=="coding":
        query = "Please code this transcript using the standard codebook."

    transcripts = []
    if uploaded_file:
        transcripts.append(uploaded_file)
    elif batch_folder and os.path.isdir(batch_folder):
        for root, dirs, files in os.walk(batch_folder):
            for f in files:
                if f.lower().endswith((".pdf",".docx",".txt")):
                    transcripts.append(os.path.join(root,f))

    if not transcripts:
        st.warning("No transcript provided!")
    else:
        for tf in transcripts:
            # 读取文本
            if hasattr(tf, "read"):
                text = load_transcript_text(tf)
                fname_base = tf.name
            else:
                with open(tf,"rb") as f:
                    text = load_transcript_text(f)
                fname_base = os.path.basename(tf)

            # profile
            profile = {"role": manual_role, "technical_level":"medium", "goal":"understanding", "short_reason":""}
            if use_resume_profile and uploaded_file:
                onboard_user_background(user_id=user_id, raw_background_inputs=[{"source_type":"resume","raw_text":text}])
                retrieved_background = retrieve_user_background(user_id=user_id, query=query, recommended_chunk_types=["knowledge_boundary","expression_preference","technical_exposure"])
                if retrieved_background.get("structured_profile"):
                    profile["role"] = retrieved_background["structured_profile"].get("role_lens",manual_role)

            # -----------------------------
            # Coding 模式
            # -----------------------------
            if mode=="coding":
                chunks = [text[i:i+2000] for i in range(0,len(text),2000)]
                aggregated_output = []
                for chunk in chunks:
                    # 获取 RAG 上下文
                    rag_result = rag.answer_question(query=chunk, mode="coding", role=profile["role"], user_profile=profile)
                    retrieved_context = rag_result.get("retrieved_context","")
                    # LLM coding with context
                    aggregated_output.extend(run_llm_coding_with_context(chunk, codebook, client, retrieved_context))

                # JSON 输出
                st.subheader(f"Coding Output (JSON) for {fname_base}")
                st.json(aggregated_output)

                # CSV 输出
                csv_file = f"{fname_base}_coding.csv"
                df = pd.DataFrame([{"segment_index":i+1,"file_name":fname_base,"text": seg["text"], "codes": ",".join(seg.get("codes",[]))} for i, seg in enumerate(aggregated_output)])
                df.to_csv(csv_file,index=False)
                st.write(f"CSV saved: {csv_file}")
                with open(csv_file,"rb") as f:
                    st.download_button(label=f"Download {csv_file}", data=f, file_name=csv_file, mime="text/csv")

                # Batch 汇总
                output_folder = batch_folder if batch_folder else "."
                summary_df, code_counts = generate_batch_summary(output_folder)
                if summary_df is not None:
                    st.subheader("Batch Code Frequency Heatmap")
                    fig, ax = plt.subplots(figsize=(10,2))
                    sns.heatmap(code_counts.to_frame().T, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)

            # -----------------------------
            # QA / Summary 模式
            # -----------------------------
            else:
                result = rag.answer_question(query=query, mode=mode, role=profile["role"], user_profile=profile)
                st.subheader("Answer")
                st.write(result.get("answer","No answer returned."))
                st.subheader("Top Citations")
                st.write(result.get("citations",[]))
                if show_context:
                    st.subheader("Retrieved Context")
                    st.text(result.get("retrieved_context",""))

            # debug
            if show_debug:
                st.subheader("Routing / Active Profile")
                st.json({"profile": profile})
