import os
import tempfile
import json
import streamlit as st
from openai import OpenAI

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


def display_citations(citations):
    if citations:
        for c in citations:
            page = f"p.{c['page']}" if c.get("page") else "doc"
            st.write(
                f"{c.get('source_file', 'unknown')} | {page} | "
                f"{c.get('section', 'unknown')} | {c.get('aim', 'unknown')} | "
                f"score={c.get('score', 'n/a')}"
            )
    else:
        st.write("No citations available.")


def build_user_profile_from_background(retrieved_background: dict) -> dict:
    structured = (retrieved_background or {}).get("structured_profile") or {}
    role = structured.get("role_lens", "general")
    if role == "product_manager":
        role = "pm"
    return {
        "role": role,
        "technical_level": structured.get("technical_depth", "medium"),
        "goal": "understanding",
        "short_reason": structured.get("short_reason", "")
    }


def answer_with_external_knowledge(query: str, user_profile=None, role="general") -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    technical_level = user_profile.get("technical_level", "medium") if user_profile else "medium"
    goal = user_profile.get("goal", "understanding") if user_profile else "understanding"
    short_reason = user_profile.get("short_reason", "") if user_profile else ""

    prompt = f"""
You are a helpful assistant.

The user is asking a general concept question.

User role: {role}
Technical level: {technical_level}
Goal: {goal}
Profile hint: {short_reason}

Instructions:
- Answer clearly and accurately using general knowledge.
- Adapt explanation to user's background.
- Role-specific emphasis:
    - business: practical meaning, workflow, value
    - pm: workflow, dependencies, deliverables
    - engineer: architecture, tradeoffs, mechanism
- Adjust jargon based on technical level.

Question:
{query}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Explain clearly and adapt to user's background."},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content.strip()
    return {"answer": answer, "citations": [], "retrieved_context": "External/general knowledge route"}


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
uploaded_file = st.sidebar.file_uploader("Upload transcript (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
use_resume_profile = st.sidebar.checkbox("Use uploaded resume to infer profile", value=True)
allow_manual_override = st.sidebar.checkbox("Allow manual role override", value=True)
user_id = st.sidebar.text_input("User ID", value="demo_user")
query = st.text_area("Enter your question or paste transcript for coding", height=140)


# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question or transcript.")
    else:
        try:
            inferred_profile = None
            effective_role = manual_role
            retrieved_background = None
            orchestration_result = None

            # Step 1: Background onboarding
            if uploaded_file and use_resume_profile:
                with st.spinner("Reading transcript and onboarding background..."):
                    transcript_text = load_transcript_text(uploaded_file)
                    onboard_user_background(user_id=user_id, raw_background_inputs=[{"source_type": "resume", "raw_text": transcript_text}])

            # Step 2: Query understanding + routing
            with st.spinner("Understanding query and planning workflow..."):
                orchestration_result = process_query(user_id=user_id, raw_query=query, has_uploaded_project_doc=True)
            query_understanding = orchestration_result["query_understanding_object"]
            routing_decision = orchestration_result["routing_decision"]

            # Step 3: Clarification route
            if routing_decision["route"] == "clarification":
                st.subheader("Clarification Needed")
                st.write(routing_decision["message"])
                if show_debug:
                    st.subheader("Query Understanding"); st.json(query_understanding)
                    st.subheader("Routing Decision"); st.json(routing_decision)

            # Step 4: Retrieval + generation
            else:
                if "background_request" in routing_decision:
                    bg_req = routing_decision["background_request"]
                    retrieved_background = retrieve_user_background(user_id=bg_req["user_id"], query=bg_req["query"], recommended_chunk_types=bg_req["recommended_background_chunk_types"])
                    if retrieved_background.get("structured_profile"):
                        inferred_profile = build_user_profile_from_background(retrieved_background)

                # Effective role determination
                if inferred_profile:
                    effective_role = manual_role if (allow_manual_override and manual_role != "general") else inferred_profile["role"]
                else:
                    effective_role = manual_role

                if show_debug:
                    st.subheader("Active Profile")
                    st.json({"effective_role_used_for_generation": effective_role, "inferred_profile": inferred_profile, "background_retrieval": retrieved_background})

                # Step 5: Generate output
                with st.spinner("Generating output..."):
                    if mode == "coding":
                        codebook = ["environmental_barrier", "social_support", "healthcare_access", "stigma", "mental_health"]
                        chunks = [query[i:i+2000] for i in range(0, len(query), 2000)]
                        aggregated_output = []
                        api_key = os.environ.get("OPENAI_API_KEY")
                        if not api_key:
                            st.error("OPENAI_API_KEY not set.")
                        else:
                            client = OpenAI(api_key=api_key)
                            for chunk in chunks:
                                prompt = f"""
You are a qualitative research coding assistant.

Transcript segment:
{chunk}

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
                                    messages=[{"role":"system","content":"Expert qualitative coding assistant."},{"role":"user","content":prompt}]
                                )
                                content = response.choices[0].message.content.strip()
                                try: parsed = json.loads(content); aggregated_output.extend(parsed)
                                except json.JSONDecodeError: aggregated_output.append({"text": chunk,"codes":["PARSE_ERROR"],"raw": content})

                        st.subheader("Coding Output (JSON)")
                        st.json(aggregated_output)
                    else:
                        route = routing_decision["route"]
                        if route == "external_knowledge_then_expression":
                            result = answer_with_external_knowledge(query=query, user_profile=inferred_profile, role=effective_role)
                        else:
                            result = rag.answer_question(query=query, mode=mode, role=effective_role, user_profile=inferred_profile)
                        st.subheader("Answer"); st.write(result["answer"])
                        st.subheader("Top Citations"); display_citations(result.get("citations", []))
                        if show_context: st.subheader("Retrieved Context"); st.text(result.get("retrieved_context", ""))

        except Exception as e:
            st.error(f"Error: {e}")
