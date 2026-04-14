import os
import tempfile
import streamlit as st
from openai import OpenAI

from rag_system import initialize_rag, load_pdf, load_docx
from background_memory import onboard_user_background, retrieve_user_background
from query_orchestrator import process_query


st.set_page_config(page_title="TechMPower RAG Assistant", layout="wide")

st.title("TechMPower RAG Assistant")
st.caption("Document-grounded RAG system with background-aware personalization")


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

    return {
        "role": structured.get("role_lens", "general"),
        "technical_level": structured.get("technical_depth", "medium"),
        "goal": "understanding",
        "short_reason": structured.get("short_reason", "")
    }


def answer_with_external_knowledge(
    query: str,
    user_profile: dict | None = None,
    role: str = "general"
) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    technical_level = "medium"
    goal = "understanding"
    short_reason = ""

    if user_profile:
        technical_level = user_profile.get("technical_level", "medium")
        goal = user_profile.get("goal", "understanding")
        short_reason = user_profile.get("short_reason", "")

    prompt = f"""
You are a helpful assistant.

The user is asking a general concept question, not a question tied to the uploaded project PDF.

User role: {role}
Technical level: {technical_level}
Goal: {goal}
Profile hint: {short_reason}

Instructions:
- Answer clearly and accurately using general knowledge.
- Adapt the explanation to the user's likely background.
- If the user is less technical, simplify jargon.
- If the user is more technical, include more mechanism/detail.
- Do not say "the evidence is insufficient."
- Do not say "human review required."
- Do not mention missing project documents.

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You explain concepts clearly and adapt explanations to the user's background."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "citations": [],
        "retrieved_context": "External/general knowledge route"
    }


if "rag" not in st.session_state:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = initialize_rag(
            docs_dir=".",
            force_rebuild=False
        )

rag = st.session_state.rag


# -----------------------------
# Sidebar
# -----------------------------
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
show_debug = st.sidebar.checkbox("Show debug info", value=True)

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

user_id = st.sidebar.text_input("User ID", value="demo_user")


# -----------------------------
# Main input
# -----------------------------
query = st.text_area("Enter your question", height=140)


# -----------------------------
# Run
# -----------------------------
if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            inferred_profile = None
            effective_role = manual_role
            retrieved_background = None
            orchestration_result = None
            query_understanding = None
            routing_decision = None

            # -----------------------------
            # Step 1: Background onboarding
            # -----------------------------
            if uploaded_file is not None and use_resume_profile:
                with st.spinner("Reading resume and onboarding user background..."):
                    resume_text = load_resume_text(uploaded_file)

                    onboard_user_background(
                        user_id=user_id,
                        raw_background_inputs=[
                            {
                                "source_type": "resume",
                                "raw_text": resume_text
                            }
                        ]
                    )

            # -----------------------------
            # Step 2: Query understanding + routing
            # -----------------------------
            with st.spinner("Understanding query and planning workflow..."):
                orchestration_result = process_query(
                    user_id=user_id,
                    raw_query=query,
                    has_uploaded_project_doc=True
                )

            query_understanding = orchestration_result["query_understanding_object"]
            routing_decision = orchestration_result["routing_decision"]

            # -----------------------------
            # Step 3: Clarification route
            # -----------------------------
            if routing_decision["route"] == "clarification":
                st.subheader("Clarification Needed")
                st.write(routing_decision["message"])

                if show_debug:
                    st.subheader("Query Understanding")
                    st.json(query_understanding)

                    st.subheader("Routing Decision")
                    st.json(routing_decision)

            # -----------------------------
            # Step 4: Non-clarification routes
            # -----------------------------
            else:
                if "background_request" in routing_decision:
                    bg_req = routing_decision["background_request"]

                    retrieved_background = retrieve_user_background(
                        user_id=bg_req["user_id"],
                        query=bg_req["query"],
                        recommended_chunk_types=bg_req["recommended_background_chunk_types"]
                    )

                    if retrieved_background.get("structured_profile") is not None:
                        inferred_profile = build_user_profile_from_background(retrieved_background)

                # Resolve final role
                if inferred_profile and not allow_manual_override:
                    effective_role = inferred_profile["role"]
                elif inferred_profile and allow_manual_override:
                    effective_role = manual_role
                else:
                    effective_role = manual_role

                if show_debug:
                    st.subheader("Query Understanding")
                    st.json(query_understanding)

                    st.subheader("Routing Decision")
                    st.json(routing_decision)

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

                # -----------------------------
                # Step 5: Generate answer
                # -----------------------------
                with st.spinner("Generating answer..."):
                    route = routing_decision["route"]

                    if route == "external_knowledge_then_expression":
                        result = answer_with_external_knowledge(
                            query=query,
                            user_profile=inferred_profile,
                            role=effective_role
                        )
                    else:
                        result = rag.answer_question(
                            query=query,
                            mode=mode,
                            role=effective_role,
                            user_profile=inferred_profile
                        )

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Top Citations")
                display_citations(result.get("citations", []))

                if show_context:
                    st.subheader("Retrieved Context")
                    st.text(result.get("retrieved_context", ""))

        except Exception as e:
            st.error(f"Error: {e}")
