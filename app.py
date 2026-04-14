import os
import tempfile
import streamlit as st

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background


st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")

st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with multi-role responses")


def load_resume_text(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            pages = load_pdf(tmp_path)
        elif suffix == ".docx":
            pages = load_docx(tmp_path)
        else:
            return ""

        text = " ".join(page_text for _, page_text in pages)
        return text[:5000]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def display_citations(citations):
    if citations:
        for c in citations:
            page = f"p.{c['page']}" if c["page"] else "doc"
            st.write(
                f"{c['source_file']} | {page} | {c['section']} | {c['aim']} | score={c['score']}"
            )
    else:
        st.write("No citations available.")


if "rag" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = initialize_rag(docs_dir=".", force_rebuild=False)

rag = st.session_state.rag

st.sidebar.header("Settings")

mode = st.sidebar.selectbox(
    "Choose mode",
    ["qa", "summary", "coding"]
)

manual_role = st.sidebar.selectbox(
    "Choose response perspective",
    ["general", "pm", "engineer", "business"]
)

show_context = st.sidebar.checkbox("Show retrieved context", value=False)

uploaded_file = st.sidebar.file_uploader(
    "Upload resume (PDF/DOCX)",
    type=["pdf", "docx"]
)

use_resume_profile = st.sidebar.checkbox(
    "Use uploaded resume to infer profile",
    value=True
)

allow_manual_override = st.sidebar.checkbox(
    "Allow manual role override",
    value=True
)

query = st.text_area("Enter your question", height=140)


if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            inferred_profile = None
            effective_role = manual_role
            retrieved_background = None

            if uploaded_file is not None and use_resume_profile:
                with st.spinner("Reading resume and onboarding user background..."):
                    resume_text = load_resume_text(uploaded_file)

                    onboard_user_background(
                        user_id="demo_user",
                        raw_background_inputs=[
                            {
                                "source_type": "resume",
                                "raw_text": resume_text
                            }
                        ]
                    )

                    retrieved_background = retrieve_user_background(
                        user_id="demo_user",
                        query=query,
                        recommended_chunk_types=[
                            "role_identity",
                            "knowledge_boundary",
                            "expression_preference",
                            "current_project",
                            "technical_exposure"
                        ]
                    )

                    structured = retrieved_background.get("structured_profile") or {}

                    inferred_profile = {
                        "role": structured.get("role_lens", "general"),
                        "technical_level": structured.get("technical_depth", "medium"),
                        "goal": "understanding",
                        "short_reason": structured.get("short_reason", "")
                    }

                if inferred_profile and not allow_manual_override:
                    effective_role = inferred_profile["role"]
                elif inferred_profile and allow_manual_override:
                    effective_role = manual_role

            st.subheader("Active Profile")
            if inferred_profile:
                st.json(
                    {
                        "inferred_profile": inferred_profile,
                        "effective_role_used_for_generation": effective_role,
                        "background_retrieval": retrieved_background
                    }
                )
            else:
                st.json(
                    {
                        "effective_role_used_for_generation": effective_role,
                        "profile_source": "manual selection"
                    }
                )

            with st.spinner("Generating answer..."):
                result = rag.answer_question(
                    query=query,
                    mode=mode,
                    role=effective_role,
                    user_profile=inferred_profile
                )

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Top Citations")
            display_citations(result["citations"])

            if show_context:
                st.subheader("Retrieved Context")
                st.text(result["retrieved_context"])

        except Exception as e:
            st.error(f"Error: {e}")
