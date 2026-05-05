import os
import re
import json
import pickle
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

try:
    from core.expression_layer import generate_personalized_explanation
except ImportError:
    try:
        from expression_layer import generate_personalized_explanation
    except ImportError:  # Allows rag_system.py to run before expression_layer.py is available
        generate_personalized_explanation = None

try:
    import faiss
except ImportError as e:
    raise ImportError("Please install faiss-cpu: pip install faiss-cpu") from e


# =========================================================
# Config
# =========================================================

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 50

BM25_TOP_K = 12
VECTOR_TOP_K = 12
RERANK_TOP_K = 8

INDEX_DIR = "techmpower_index"
DEFAULT_DOCS_DIR = "."

USE_OPENAI = True

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_FALLBACK_MODEL = "gpt-4o-mini"


# =========================================================
# Data structure
# =========================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    source_type: str
    page: Optional[int]
    section: str
    aim: str
    data_type: str
    sensitivity: str
    human_review_required: bool


# =========================================================
# Guardrails
# =========================================================

BLOCKED_PATTERNS = [
    r"\beligibility\b",
    r"\bparole\b",
    r"\bcustody\b",
    r"\brisk score\b",
    r"\brisk prediction\b",
    r"\bwhich participant\b",
    r"\bwho should be prioritized\b",
    r"\bclinical decision\b",
    r"\bshould this person receive\b",
    r"\bcriminal behavior\b",
    r"\bsurveillance\b",
]

GUARDRAIL_MESSAGE = (
    "This system is limited to analytic augmentation and document-supported research assistance. "
    "It cannot make participant-level clinical, legal, eligibility, or custody-related decisions. "
    "Please refer the case for trained human review."
)


ROLE_PROMPTS = {
    "general": (
        "You are a careful research assistant. "
        "Answer clearly, accurately, and only from the retrieved evidence."
    ),
    "pm": (
        "You are answering as a Product Manager. "
        "Focus on user needs, workflow impact, tradeoffs, implementation feasibility, "
        "and what should happen next operationally."
    ),
    "engineer": (
        "You are answering as an Engineer. "
        "Focus on system design, technical implementation, architecture, data flow, "
        "risks, constraints, and concrete build details."
    ),
    "business": (
        "You are answering as a Business Manager. "
        "Focus on business value, stakeholder impact, scalability, cost, adoption, "
        "operational efficiency, and strategic implications."
    ),
}

# =========================================================
# General source-of-truth grounding policy for any project

SOURCE_OF_TRUTH_POLICY = (
    "Source-of-truth policy: The uploaded/retrieved document is the only source of truth. "
    "Do not use outside knowledge associated with project names, brands, datasets, papers, frameworks, companies, public figures, books, films, or famous cases unless that information appears in the retrieved context. "
    "Do not infer real-world deployment, business impact, ROI, cost savings, customer impact, model success, selected model, causal conclusions, or production readiness unless explicitly supported by the retrieved evidence. "
    "For business-facing answers, clearly distinguish documented outcomes from potential value. Use cautious wording such as 'could support' only when grounded in the project objective, and do not convert potential value into proven impact. "
    "If multiple candidate models appear in the retrieved evidence, do not claim a selected/final/best model unless the retrieved context explicitly says selected, best, final, chosen, deployed, recommended, lowest AIC/BIC/SBC, highest validation performance, or conclusion. "
    "If a detail is not specified in the retrieved evidence, say that it is not specified rather than filling the gap with general knowledge."
)

PROJECT_ANALYSIS_REASONING_RUBRIC = (
    "Project-analysis reasoning rubric: Before writing the final answer, silently analyze the uploaded project through these lenses: "
    "1) artifact identity: decide whether the document is a modeling report, software README, dashboard, portfolio website, research proposal, notebook, or mixed project portfolio; "
    "2) project objective: identify the actual business, analytical, technical, or research goal; "
    "3) end-to-end workflow: reconstruct the concrete pipeline from inputs to outputs, including data collection, preprocessing, feature engineering, modeling, evaluation, deployment, or user-facing components when present; "
    "4) model/system roles: distinguish causal inference models, forecasting models, retrieval models, embedding models, UI components, data pipelines, and evaluation layers instead of collapsing them into one generic model; "
    "5) evidence-backed results: extract actual metrics, tables, charts, selected-model statements, and failure modes when available; "
    "6) limitations: identify what is weak, missing, uncertain, under-validated, noisy, biased, or not production-ready; "
    "7) next engineering actions: propose concrete reproduction, validation, monitoring, debugging, or improvement steps grounded in the document. "
    "Use this rubric to organize the answer, but do not reveal hidden reasoning or internal chain-of-thought. Return only the final explanation."
)


# =========================================================
# Profile-aware prompt builder
# =========================================================

def build_profile_prompt(role: str = "general", user_profile: Optional[Dict] = None) -> str:
    role = (role or "general").lower()
    role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["general"])

    if not user_profile:
        return role_prompt

    technical_level = user_profile.get("technical_level", "medium")
    goal = user_profile.get("goal", "understanding")
    short_reason = user_profile.get("short_reason", "")

    level_instruction_map = {
        "low": (
            "Use simpler language, reduce jargon, define technical terms briefly, "
            "and explain ideas step by step."
        ),
        "medium": (
            "Use moderately technical language, but keep the explanation clear and structured."
        ),
        "high": (
            "Use more technical depth, include implementation details, tradeoffs, "
            "and domain-specific terminology when appropriate."
        ),
    }

    goal_instruction_map = {
        "understanding": (
            "Prioritize conceptual clarity and explain what the evidence means."
        ),
        "decision": (
            "Prioritize implications, risks, tradeoffs, and what decision-makers should consider next."
        ),
        "implementation": (
            "Prioritize operational steps, system design, execution details, and implementation constraints."
        ),
    }

    level_instruction = level_instruction_map.get(technical_level, level_instruction_map["medium"])
    goal_instruction = goal_instruction_map.get(goal, goal_instruction_map["understanding"])

    profile_prompt = (
        f"{role_prompt}\n\n"
        "Additional user profile guidance:\n"
        f"- Technical level: {technical_level}\n"
        f"- Goal: {goal}\n"
        f"- Inference note: {short_reason}\n\n"
        "Adapt the answer accordingly:\n"
        f"- {level_instruction}\n"
        f"- {goal_instruction}\n"
        "- Keep the answer faithful to the retrieved evidence.\n"
        "- Do not invent facts beyond the documents.\n"
        "- Do not add external examples, business outcomes, deployment claims, or famous-case background that are not in the retrieved evidence."
    )

    return profile_prompt


# =========================================================
# Helper: Expression Layer compatibility

def build_basic_query_understanding(query: str, mode: str = "qa") -> Dict:
    """Create a lightweight query understanding object for the expression layer.

    The full query understanding still lives in query_orchestrator.py. This helper
    keeps rag_system.py compatible when it is called directly.
    """
    q = (query or "").lower()

    project_terms = [
        "project", "case study", "report", "analysis", "modeling project",
        "what does this project do", "tell me about", "walk me through", "overview"
    ]

    if " vs " in q or "difference between" in q or "compare" in q:
        query_type = "comparison_question"
        intent = "compare_options"
    elif any(term in q for term in project_terms) or mode == "summary":
        query_type = "project_explanation"
        intent = "summarize_project_evidence"
    elif "workflow" in q or "pipeline" in q or "how does" in q or "architecture" in q:
        query_type = "workflow_explanation"
        intent = "understand_process"
    elif "document" in q or "note" in q or "based on" in q:
        query_type = "document_based_question"
        intent = "understand_document_context"
    else:
        query_type = "concept_explanation"
        intent = "understand_concept"

    topic = query.strip().rstrip("?")[:120] if query else "unknown"

    return {
        "query_type": query_type,
        "topic": topic,
        "intent": intent,
        "domain": "retrieved project documents",
        "requires_background_retrieval": True,
        "requires_project_context": True,
        "needs_clarification": False,
    }


def build_expression_background_package(
    user_profile: Optional[Dict] = None,
    retrieved_background_package: Optional[Dict] = None,
) -> Dict:
    """Normalize background/profile data into the shape expected by expression_layer.py."""
    if retrieved_background_package:
        return retrieved_background_package

    if not user_profile:
        return {}

    structured_profile = {
        "role_lens": user_profile.get("role_lens") or user_profile.get("role"),
        "technical_depth": user_profile.get("technical_depth") or user_profile.get("technical_level"),
        "technical_level": user_profile.get("technical_level"),
        "jargon_tolerance": user_profile.get("jargon_tolerance", "medium"),
        "preferred_explanation_style": user_profile.get("preferred_explanation_style", []),
        "goal": user_profile.get("goal", "understanding"),
        "short_reason": user_profile.get("short_reason", ""),
    }

    background_chunks = []
    if user_profile.get("short_reason"):
        background_chunks.append({
            "chunk_type": "profile_inference",
            "text": user_profile["short_reason"],
        })
    if user_profile.get("weak_areas"):
        background_chunks.append({
            "chunk_type": "knowledge_boundary",
            "text": "Weak areas: " + ", ".join(user_profile.get("weak_areas", [])),
        })
    if user_profile.get("preferred_explanation_style"):
        styles = user_profile.get("preferred_explanation_style", [])
        if isinstance(styles, list):
            styles_text = ", ".join(styles)
        else:
            styles_text = str(styles)
        background_chunks.append({
            "chunk_type": "expression_preference",
            "text": "Preferred explanation style: " + styles_text,
        })

    return {
        "structured_profile": structured_profile,
        "retrieved_background_chunks": background_chunks,
    }


# =========================================================
# Utility functions
# =========================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_/-]+", text.lower())


def detect_source_type(filename: str) -> str:
    low = filename.lower()
    if "workflow" in low:
        return "Workflow"
    if "appendix" in low or "ai" in low or "llm" in low:
        return "AI_Appendix"
    if "datasheet" in low or "irb" in low or "protocol" in low:
        return "IRB_Protocol"
    return "Other"


# =========================================================
# Project/Resume Document Kind and Chunking Helpers
# =========================================================

def detect_document_kind(filename: str) -> str:
    """Classify uploaded documents for chunking and metadata behavior."""
    low = filename.lower()
    if "resume" in low or "cv" in low or "curriculum_vitae" in low:
        return "resume"
    if any(term in low for term in ["project", "report", "case", "analysis", "assignment", "paper", "portfolio", "readme"]):
        return "project"
    return "project"


def infer_project_section(text: str, fallback: str = "Project Evidence") -> str:
    """Infer a general project/report section without TechMPower-specific labels."""
    low = text.lower()
    if any(term in low for term in ["abstract", "executive summary", "summary"]):
        return "Summary"
    if any(term in low for term in ["introduction", "motivation", "background", "problem statement", "objective", "goal"]):
        return "Project Objective / Background"
    if any(term in low for term in ["data", "dataset", "data source", "variables", "target variable", "sample", "time period"]):
        return "Dataset / Variables"
    if any(term in low for term in ["data preparation", "preprocessing", "cleaning", "missing", "imputation", "outlier", "transformation", "feature"]):
        return "Data Preparation / Feature Engineering"
    if any(term in low for term in ["methodology", "method", "model", "algorithm", "regression", "classification", "forecast", "training"]):
        return "Modeling Methodology"
    if any(term in low for term in ["model selection", "selected model", "best model", "aic", "bic", "sbc", "rmse", "mae", "accuracy", "auc", "r-squared", "adjusted r"]):
        return "Model Selection / Evaluation"
    if any(term in low for term in ["result", "performance", "findings", "evaluation", "table", "metric"]):
        return "Results / Findings"
    if any(term in low for term in ["limitation", "risk", "caveat", "future work", "next step", "conclusion", "recommendation"]):
        return "Limitations / Conclusion"
    return fallback


def infer_project_aim(text: str) -> str:
    """General evidence type for project/report documents."""
    low = text.lower()
    if any(term in low for term in ["objective", "goal", "purpose", "problem statement"]):
        return "Objective Evidence"
    if any(term in low for term in ["dataset", "data", "variables", "target", "sample"]):
        return "Data Evidence"
    if any(term in low for term in ["model", "method", "algorithm", "regression", "classification", "forecast"]):
        return "Method Evidence"
    if any(term in low for term in ["selected", "best model", "aic", "bic", "sbc", "rmse", "mae", "accuracy", "auc", "r-squared"]):
        return "Model Selection Evidence"
    if any(term in low for term in ["result", "performance", "metric", "finding"]):
        return "Result Evidence"
    if any(term in low for term in ["limitation", "risk", "future work", "conclusion", "recommendation"]):
        return "Limitation / Next-Step Evidence"
    return "Project Evidence"


def split_project_text_by_paragraph_or_heading(text: str) -> List[str]:
    """Chunk project documents by paragraph/heading-like boundaries instead of fixed words only."""
    raw_blocks = []
    current = []

    # Preserve likely paragraph/heading boundaries before clean_text collapses whitespace.
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                raw_blocks.append(" ".join(current).strip())
                current = []
            continue

        is_heading = (
            len(stripped) <= 90
            and (
                stripped.isupper()
                or re.match(r"^(\d+\.?\s+|[A-Z][A-Za-z ]+:$|#{1,6}\s+)", stripped)
                or stripped.lower() in {
                    "introduction", "background", "methodology", "methods", "data", "dataset",
                    "data preparation", "modeling", "model selection", "results", "conclusion",
                    "limitations", "future work", "recommendations"
                }
            )
        )
        if is_heading and current:
            raw_blocks.append(" ".join(current).strip())
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        raw_blocks.append(" ".join(current).strip())

    blocks = [clean_text(block) for block in raw_blocks if clean_text(block)]
    if not blocks:
        blocks = split_into_sentential_units(text)

    merged_chunks = []
    buffer = ""
    for block in blocks:
        candidate = (buffer + " " + block).strip() if buffer else block
        if len(candidate.split()) <= CHUNK_SIZE_WORDS:
            buffer = candidate
        else:
            if buffer:
                merged_chunks.append(buffer)
            if len(block.split()) > CHUNK_SIZE_WORDS:
                merged_chunks.extend(chunk_text(block))
                buffer = ""
            else:
                buffer = block

    if buffer:
        merged_chunks.append(buffer)

    return merged_chunks


def infer_section(text: str, source_type: str) -> str:
    low = text.lower()
    if "study design" in low or "hybrid type ii" in low or "stepped wedge" in low:
        return "Study Design"
    if "privacy" in low or "data security" in low or "hipaa" in low:
        return "Privacy & Data Security"
    if "aim 1" in low or "effectiveness" in low:
        return "Aim 1"
    if "aim 2" in low or "implementation" in low or "prism" in low or "re-aim" in low:
        return "Aim 2"
    if "aim 3" in low or "cost-effectiveness" in low or "sustainability" in low:
        return "Aim 3"
    if "workflow" in low or "human in the loop" in low:
        return "Workflow"
    return source_type


def infer_aim(text: str) -> str:
    low = text.lower()
    if "aim 1" in low or "effectiveness" in low:
        return "Aim 1"
    if "aim 2" in low or "implementation" in low or "prism" in low or "re-aim" in low:
        return "Aim 2"
    if "aim 3" in low or "cost-effectiveness" in low or "sustainability" in low:
        return "Aim 3"
    return "Cross-cutting"


def infer_data_type(text: str) -> str:
    low = text.lower()
    if "acasi" in low or "survey" in low:
        return "Survey"
    if "interview" in low or "qualitative" in low or "transcript" in low:
        return "Qualitative Text"
    if "emr" in low or "medical record" in low:
        return "Medical Records"
    if "training" in low or "attendance" in low or "implementation log" in low:
        return "Training/Implementation"
    if "cost" in low or "staff time" in low:
        return "Cost Data"
    if "workflow" in low:
        return "Workflow"
    return "General Research Text"


def infer_sensitivity(text: str) -> str:
    low = text.lower()
    if "phi" in low or "pii" in low or "hipaa" in low or "medical record" in low:
        return "High"
    if "de-identified" in low or "aggregate" in low:
        return "Medium"
    return "Low"


def split_into_sentential_units(text: str) -> List[str]:
    pieces = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z])", text)
    return [p.strip() for p in pieces if p.strip()]


def chunk_text(
    text: str,
    chunk_size_words: int = CHUNK_SIZE_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS
) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


# =========================================================
# File loading
# =========================================================

def load_pdf(filepath: str) -> List[Tuple[int, str]]:
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Keep raw line breaks here so project chunking can use paragraph/heading structure.
        text = text.replace("\x00", " ").strip()
        if text:
            pages.append((i + 1, text))
    return pages


def load_docx(filepath: str) -> List[Tuple[Optional[int], str]]:
    doc = DocxDocument(filepath)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    text = "\n\n".join(paragraphs)
    return [(None, text)] if text else []


def load_document(filepath: str) -> List[Tuple[Optional[int], str]]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return load_pdf(filepath)
    elif ext == ".docx":
        return load_docx(filepath)
    elif ext in {".md", ".txt"}:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().replace("\x00", " ").strip()
        return [(None, text)] if text else []
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def build_chunks_for_file(filepath: str) -> List[Chunk]:
    filename = os.path.basename(filepath)
    source_type = detect_source_type(filename)
    document_kind = detect_document_kind(filename)
    raw_units = load_document(filepath)

    output = []
    counter = 0

    if document_kind == "resume":
        full_text = clean_text("\n\n".join(raw_text for _, raw_text in raw_units))
        if not full_text:
            return []
        return [
            Chunk(
                chunk_id=f"{filename}_resume_full",
                text=full_text,
                source_file=filename,
                source_type="Resume",
                page=None,
                section="Resume Full Document",
                aim="Profile Evidence",
                data_type="Resume / Profile Text",
                sensitivity="Medium",
                human_review_required=True,
            )
        ]

    for page_num, raw_text in raw_units:
        subchunks = split_project_text_by_paragraph_or_heading(raw_text)

        for sub in subchunks:
            section = infer_project_section(sub, fallback=source_type)
            chunk = Chunk(
                chunk_id=f"{filename}_p{page_num or 0}_c{counter}",
                text=clean_text(sub),
                source_file=filename,
                source_type="Project Document",
                page=page_num,
                section=section,
                aim=infer_project_aim(sub),
                data_type=infer_data_type(sub),
                sensitivity=infer_sensitivity(sub),
                human_review_required=True,
            )
            output.append(chunk)
            counter += 1

    return output


# =========================================================
# Main RAG class
# =========================================================

class TechMPowerRAG:
    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME
    ):
        self.embed_model_name = embed_model_name
        self.reranker_model_name = reranker_model_name

        # Force CPU to avoid PyTorch meta-tensor / MPS device issues on some Mac environments.
        self.embed_model = SentenceTransformer(self.embed_model_name, device="cpu")
        self.reranker = CrossEncoder(self.reranker_model_name, device="cpu")

        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        self.bm25 = None
        self.bm25_tokens = None
        self.embeddings = None
        self.index = None

    def _blocked_query(self, query: str) -> bool:
        q = query.lower()
        return any(re.search(p, q) for p in BLOCKED_PATTERNS)

    def _discover_files(self, docs_dir: str) -> List[str]:
        files = []
        for fname in os.listdir(docs_dir):
            if fname.lower().endswith((".pdf", ".docx", ".md", ".txt")):
                files.append(os.path.join(docs_dir, fname))
        return sorted(files)

    def build_index(self, docs_dir: str = DEFAULT_DOCS_DIR) -> None:
        files = self._discover_files(docs_dir)
        if not files:
            raise FileNotFoundError(
                f"No PDF/DOCX/MD/TXT files found in '{docs_dir}'. "
                "In Colab, your uploaded files are usually in the current directory '.', "
                "so use rag.build_index('.')"
            )

        all_chunks = []
        for fp in files:
            print(f"Processing: {os.path.basename(fp)}")
            all_chunks.extend(build_chunks_for_file(fp))

        if not all_chunks:
            raise ValueError("Files were found, but no text could be extracted.")

        self.chunks = all_chunks
        self.chunk_texts = [c.text for c in self.chunks]

        self.bm25_tokens = [tokenize_for_bm25(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(self.bm25_tokens)

        print("Encoding dense embeddings...")
        embeddings = self.embed_model.encode(
            self.chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.embeddings = embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.save(INDEX_DIR)
        print(f"Index built and saved to '{INDEX_DIR}'.")

    def save(self, out_dir: str = INDEX_DIR) -> None:
        ensure_dir(out_dir)

        with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        with open(os.path.join(out_dir, "bm25_tokens.pkl"), "wb") as f:
            pickle.dump(self.bm25_tokens, f)

        np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)
        faiss.write_index(self.index, os.path.join(out_dir, "faiss.index"))

    def load(self, out_dir: str = INDEX_DIR) -> None:
        with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

        with open(os.path.join(out_dir, "bm25_tokens.pkl"), "rb") as f:
            self.bm25_tokens = pickle.load(f)

        self.chunk_texts = [c.text for c in self.chunks]
        self.bm25 = BM25Okapi(self.bm25_tokens)
        self.embeddings = np.load(os.path.join(out_dir, "embeddings.npy")).astype("float32")
        self.index = faiss.read_index(os.path.join(out_dir, "faiss.index"))

        print(f"Loaded existing index from '{out_dir}'.")

    def retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> List[Tuple[Chunk, float]]:
        if self._blocked_query(query):
            raise PermissionError(GUARDRAIL_MESSAGE)

        q_tokens = tokenize_for_bm25(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(bm25_scores)[::-1][:BM25_TOP_K].tolist()

        q_emb = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, idxs = self.index.search(q_emb, VECTOR_TOP_K)
        dense_idx = idxs[0].tolist()

        candidates = sorted(set(bm25_idx + dense_idx))

        pairs = [(query, self.chunk_texts[i]) for i in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked = sorted(
            [(self.chunks[i], float(score)) for i, score in zip(candidates, rerank_scores)],
            key=lambda x: x[1],
            reverse=True
        )

        return reranked[:top_k]

    def format_context(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        blocks = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            loc = f"p.{chunk.page}" if chunk.page else "doc"
            block = (
                f"[Citation {rank} | score={score:.4f}]\n"
                f"Source: {chunk.source_file} | {loc}\n"
                f"Section: {chunk.section}\n"
                f"Aim: {chunk.aim}\n"
                f"Data Type: {chunk.data_type}\n"
                f"Text: {chunk.text}\n"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def summarize_retrieved_sources(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        """Create a compact retrieval diagnostic for the generation prompt."""
        if not retrieved:
            return "No retrieved chunks."

        by_source: Dict[str, List[int]] = {}
        for chunk, _ in retrieved:
            by_source.setdefault(chunk.source_file, [])
            if chunk.page is not None and chunk.page not in by_source[chunk.source_file]:
                by_source[chunk.source_file].append(chunk.page)

        parts = []
        for source, pages in by_source.items():
            pages_text = ", ".join(f"p.{p}" for p in sorted(pages)) if pages else "document-level text"
            parts.append(f"{source}: {pages_text}")
        return "; ".join(parts)

    def _generate_with_openai(
        self,
        query: str,
        context: str,
        mode: str,
        role: str = "general",
        user_profile: Optional[Dict] = None
    ) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Please run: "
                "os.environ['OPENAI_API_KEY'] = 'your_key'"
            )

        client = OpenAI(api_key=api_key)

        role = (role or "general").lower()
        role_prompt = build_profile_prompt(role=role, user_profile=user_profile)

        if mode == "qa":
            task_prompt = (
                "You are answering questions about uploaded project documents. "
                "Use only the retrieved evidence. "
                "If evidence is insufficient, say so clearly. "
                "When the user asks what a project does, explain it as a real project, not as a generic system template. "
                "Extract concrete details whenever available: exact project title, objective, artifact type, dataset, sample size or time period, target variable, preprocessing choices, feature engineering, full workflow steps, model names, model roles, system components, selected model, model-selection criteria, evaluation metrics, charts/tables, failure modes, limitations, and next steps. "
                "If the retrieved evidence includes numeric metrics such as R², RMSE, MAE, MSE, AIC, BIC, accuracy, AUC, p-values, posterior probabilities, impact strength, or duration, include the actual numbers and briefly explain what they imply. "
                "If different models or modules serve different parts of the project, distinguish their roles clearly, for example causal inference versus forecasting, retrieval versus generation, frontend versus backend, or data pipeline versus model evaluation. "
                "If the retrieved context appears to contain multiple projects or assignments, do not silently merge them. Identify that multiple projects may be present and clearly state which project you are summarizing based on the retrieved evidence. "
                "Do not invent software modules, business value, ROI, cost savings, customer impact, user impact, model metrics, selected-model claims, real-world outcomes, or deployment claims that are not supported by the retrieved context. "
                "If the project title resembles a famous case, company, dataset, framework, paper, book, or film, do not import that outside background unless it appears in the retrieved context. "
                "Do not force the answer into generic software modules unless the user explicitly asks for architecture or implementation. "
                "Cite evidence inline using the retrieved citation labels such as [Citation 1] or [Citation 2]."
            )
        elif mode == "summary":
            task_prompt = (
                "You are summarizing uploaded project documents. "
                "Use only the retrieved evidence. "
                "If the document contains multiple projects or assignments, identify that clearly and do not mix details across projects. "
                "Summarize concrete project details: exact project title, objective, artifact type, dataset, sample size or time period, target variable, data preparation, full workflow, modeling or system components, model roles, model-selection criteria, selected model, quantitative results, charts/tables, limitations, and next steps. "
                "If numeric metrics or evaluation results are available, include the actual numbers and interpret whether they indicate strong performance, weak performance, overfitting, underfitting, noisy signals, or validation gaps. "
                "If any detail is not supported by the retrieved context, say it is not specified in the retrieved evidence. "
                "Do not import outside knowledge from famous project names, brands, datasets, papers, frameworks, companies, books, films, or public cases. "
                "Distinguish documented outcomes from potential value; do not claim actual business impact, deployment, ROI, cost savings, or real-world success unless explicitly supported by the retrieved evidence. "
                "Do not use generic module labels unless the retrieved document explicitly uses them. "
                "Cite evidence inline using the retrieved citation labels such as [Citation 1] or [Citation 2]. "
                "Flag uncertainty clearly."
            )
        elif mode == "coding":
            task_prompt = (
                "You are assisting with first-pass qualitative coding for implementation research. "
                "Given the evidence, suggest 1-3 candidate codes and possible PRISM/RE-AIM mapping. "
                "Do not claim final coding certainty. "
                "End with: 'Final coding requires human review.'"
            )
        else:
            task_prompt = "Use only the retrieved evidence."

        system_prompt = (
            f"{role_prompt}\n\n"
            f"{task_prompt}\n\n"
            f"{SOURCE_OF_TRUTH_POLICY}\n\n"
            f"{PROJECT_ANALYSIS_REASONING_RUBRIC}\n\n"
            "Do not use outside knowledge. "
            "If the retrieved context does not support the answer, say that the evidence is insufficient. "
            "When making a specific claim from the documents, include the matching citation label from the retrieved context. "
            "The final answer should be an explanation of the uploaded project evidence, not a generic explanation of the project title or topic."
        )

        user_prompt = f"""
Question:
{query}

Retrieved source/page summary:
{self._current_retrieval_summary if hasattr(self, "_current_retrieval_summary") else "Not available"}

Retrieved context:
{context}

Silent project-analysis checklist before answering:
1. What artifact am I explaining: modeling report, software README, dashboard, portfolio website, research proposal, notebook, or mixed portfolio?
2. What is the real project objective, not just the document title?
3. What is the concrete end-to-end workflow from inputs to outputs?
4. Which models/modules serve different roles, and should they be separated in the answer?
5. What exact metrics, charts, tables, thresholds, selected-model statements, or failure modes are supported by the retrieved context?
6. What limitations, validation gaps, noisy signals, leakage risks, or production-readiness issues are documented?
7. Is each concrete claim supported by retrieved context, without importing outside knowledge?
8. Am I separating documented results from potential value or future work?
9. If model selection is mentioned, does the retrieved evidence explicitly support the selected/final/best model claim?
Return only the final answer. Do not show this checklist.
"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=6000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        if content:
            return content

        # GPT-5-family models can sometimes spend the completion budget on
        # internal reasoning and return no visible text. Retry once with a
        # stricter visible-output instruction before passing an empty base
        # explanation into the expression layer.
        retry_user_prompt = f"""
{user_prompt}

The previous response returned no visible content. Now return a concise but complete final answer in visible text.
Do not return an empty response.
Use 7 sections at most.
Include concrete workflow, model/module roles, metrics, limitations, and engineering next steps when supported by the retrieved context.
"""

        retry_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_completion_tokens=8000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_prompt},
            ],
        )

        retry_content = (retry_response.choices[0].message.content or "").strip()
        if retry_content:
            return retry_content

        # Final fallback: use a chat-optimized model if the primary GPT-5.5 call
        # still returns no visible content. This keeps the app usable instead of
        # sending an empty base explanation to the expression layer.
        fallback_response = client.chat.completions.create(
            model=OPENAI_FALLBACK_MODEL,
            max_completion_tokens=6000,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_user_prompt},
            ],
        )

        fallback_content = (fallback_response.choices[0].message.content or "").strip()
        if fallback_content:
            return fallback_content

        raise RuntimeError(
            f"Both {OPENAI_MODEL} and {OPENAI_FALLBACK_MODEL} returned empty base explanations. "
            "Use a chat model such as gpt-4o for OPENAI_MODEL or inspect the raw API response."
        )

    def _heuristic_answer(
        self,
        query: str,
        retrieved: List[Tuple[Chunk, float]],
        mode: str,
        role: str = "general"
    ) -> str:
        combined = " ".join(chunk.text for chunk, _ in retrieved)
        low = combined.lower()

        role = (role or "general").lower()
        role_prefix = {
            "pm": "Product perspective",
            "engineer": "Engineering perspective",
            "business": "Business perspective",
            "general": "General research perspective"
        }.get(role, "General research perspective")

        if mode == "summary":
            return (
                f"{role_prefix} summary:\n"
                f"{combined[:1200]}...\n\n"
                "Human review required before using this output in reporting."
            )

        if mode == "coding":
            codes = []
            if "stigma" in low:
                codes.append("Stigma / negative attitudes")
            if "training" in low:
                codes.append("Training / workforce preparation")
            if "coordination" in low or "linkage" in low:
                codes.append("Care coordination / linkage")
            if "cost" in low:
                codes.append("Cost / sustainability")
            if "technology" in low or "telehealth" in low:
                codes.append("Technology-enabled implementation")
            if "fidelity" in low:
                codes.append("Implementation fidelity")

            if not codes:
                codes = ["Implementation process", "Contextual barrier", "Needs human review"]

            return (
                f"{role_prefix} first-pass coding suggestion:\n"
                f"- Candidate codes: {', '.join(codes[:5])}\n"
                "- These are preliminary analytic suggestions only.\n"
                "- Final coding requires human review."
            )

        return (
            f"{role_prefix} answer based on retrieved study documents:\n"
            f"{combined[:1200]}...\n\n"
            "This answer is grounded in the retrieved protocol/workflow materials. "
            "Human review is required for final interpretation."
        )

    def answer_question(
        self,
        query: str,
        mode: str = "qa",
        role: str = "general",
        user_profile: Optional[Dict] = None,
        query_understanding: Optional[Dict] = None,
        retrieved_background_package: Optional[Dict] = None,
        apply_expression_layer: bool = True,
        retrieval_query: Optional[str] = None,
    ) -> Dict:
        if mode not in {"qa", "summary", "coding"}:
            raise ValueError("Mode must be one of: 'qa', 'summary', 'coding'.")

        if role not in {"general", "pm", "engineer", "business"}:
            raise ValueError("Role must be one of: 'general', 'pm', 'engineer', 'business'.")

        effective_retrieval_query = retrieval_query or query

        if self._blocked_query(query) or self._blocked_query(effective_retrieval_query):
            return {
                "mode": mode,
                "role": role,
                "query": query,
                "retrieval_query": effective_retrieval_query,
                "blocked": True,
                "answer": GUARDRAIL_MESSAGE,
                "citations": [],
                "user_profile": user_profile
            }

        retrieved = self.retrieve(effective_retrieval_query, top_k=RERANK_TOP_K)
        context = self.format_context(retrieved)
        self._current_retrieval_summary = self.summarize_retrieved_sources(retrieved)

        # Step 1: generate a neutral, evidence-grounded base explanation.
        # Personalization is handled explicitly by expression_layer.py below.
        if USE_OPENAI:
            base_explanation = self._generate_with_openai(
                query=query,
                context=context,
                mode=mode,
                role="general",
                user_profile=None
            )
        else:
            base_explanation = self._heuristic_answer(query, retrieved, mode, role="general")

        if not (base_explanation or "").strip():
            raise RuntimeError(
                "RAG base explanation is empty. Refusing to call the expression layer with empty input."
            )

        answer = base_explanation
        expression_plan = None

        # Step 2: rewrite the base explanation using the explicit Expression Layer.
        if apply_expression_layer and generate_personalized_explanation is not None:
            effective_query_understanding = query_understanding or build_basic_query_understanding(
                query=query,
                mode=mode,
            )
            expression_background_package = build_expression_background_package(
                user_profile=user_profile,
                retrieved_background_package=retrieved_background_package,
            )

            expression_result = generate_personalized_explanation(
                base_explanation=base_explanation,
                query_understanding=effective_query_understanding,
                retrieved_background_package=expression_background_package,
                role=role,
            )
            expression_plan = expression_result.get("expression_plan")
            answer = expression_result.get("final_explanation", base_explanation)

        citations = [
            {
                "citation_id": i,
                "source_file": chunk.source_file,
                "page": chunk.page,
                "section": chunk.section,
                "aim": chunk.aim,
                "data_type": chunk.data_type,
                "score": round(score, 4),
                "snippet": chunk.text[:450] + ("..." if len(chunk.text) > 450 else "")
            }
            for i, (chunk, score) in enumerate(retrieved, start=1)
        ]

        return {
            "mode": mode,
            "role": role,
            "query": query,
            "retrieval_query": effective_retrieval_query,
            "blocked": False,
            "answer": answer,
            "base_explanation": base_explanation,
            "expression_plan": expression_plan,
            "citations": citations,
            "retrieved_context": context,
            "user_profile": user_profile
        }


# =========================================================
# Convenience helpers for Colab
# =========================================================

def print_answer(result: Dict) -> None:
    print("=" * 90)
    print("MODE:", result["mode"])
    print("ROLE:", result.get("role", "general"))
    print("QUERY:", result["query"])
    print("\nANSWER:\n")
    print(result["answer"])

    print("\nTOP CITATIONS:")
    if not result["citations"]:
        print("No citations.")
        return

    for i, c in enumerate(result["citations"], start=1):
        loc = f"p.{c['page']}" if c["page"] else "doc"
        print(f"[{i}] {c['source_file']} | {loc} | {c['section']} | {c['aim']} | score={c['score']}")
        if c.get("snippet"):
            print(f"    Evidence: {c['snippet']}")


def list_uploaded_docs(docs_dir: str = ".") -> List[str]:
    files = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith((".pdf", ".docx", ".md", ".txt")):
            files.append(fname)
    return files


def move_docs_to_folder(file_list: List[str], target_dir: str = "docs") -> None:
    ensure_dir(target_dir)
    for f in file_list:
        if os.path.exists(f):
            shutil.move(f, os.path.join(target_dir, f))
            print(f"Moved: {f} -> {target_dir}/{f}")
        else:
            print(f"Not found: {f}")


def initialize_rag(docs_dir: str = ".", force_rebuild: bool = False) -> TechMPowerRAG:
    rag = TechMPowerRAG()

    index_path = os.path.join(INDEX_DIR, "faiss.index")
    if os.path.exists(index_path) and not force_rebuild:
        rag.load(INDEX_DIR)
    else:
        rag.build_index(docs_dir)

    return rag


# =========================================================
# Evaluation
# =========================================================

def make_sample_eval_questions() -> List[Dict]:
    """General project-document evaluation questions.

    `gold_concepts` are intentionally flexible. A concept is counted as covered if
    the answer contains any one of the acceptable variants for that concept.
    This avoids overfitting evaluation to exact keywords from one project.
    """
    return [
        {
            "question": "Summarize the uploaded project document and identify the main project or projects inside it.",
            "gold_concepts": {
                "project_identity": ["project", "study", "analysis", "report", "case", "assignment"],
                "objective": ["objective", "goal", "purpose", "aim", "tries to", "designed to", "intended to"],
                "data": ["dataset", "data", "sample", "observations", "records", "variables", "features"],
                "method": ["model", "method", "algorithm", "approach", "methodology", "regression", "classification", "forecast"],
                "result": ["result", "finding", "performance", "metric", "selected", "best", "output"],
                "limitation": ["limitation", "risk", "caveat", "weakness", "constraint", "future work", "next step"],
            },
        },
        {
            "question": "What is the objective of the project described in the uploaded document?",
            "gold_concepts": {
                "objective": ["objective", "goal", "purpose", "aim", "tries to", "designed to", "intended to"],
                "task": ["predict", "classify", "estimate", "analyze", "forecast", "evaluate", "compare", "understand", "identify"],
                "target_or_outcome": ["target", "outcome", "dependent variable", "response", "label", "metric", "result"],
            },
        },
        {
            "question": "What dataset and target variable does the project use?",
            "gold_concepts": {
                "data_source": ["dataset", "data", "source", "sample", "observations", "records"],
                "variables": ["variables", "features", "predictors", "columns", "inputs", "covariates"],
                "target": ["target", "outcome", "dependent variable", "response variable", "label", "y variable"],
            },
        },
        {
            "question": "What modeling methods or algorithms are used in the project?",
            "gold_concepts": {
                "method": ["model", "method", "algorithm", "approach", "methodology"],
                "model_family": ["regression", "classification", "tree", "forest", "boosting", "neural", "time series", "forecast", "clustering", "optimization", "ols", "logistic"],
                "workflow": ["train", "fit", "selection", "compare", "evaluate", "validation", "feature", "preprocess"],
            },
        },
        {
            "question": "What are the key results, limitations, and next steps for the project?",
            "gold_concepts": {
                "result": ["result", "finding", "performance", "metric", "selected", "best", "improved", "output"],
                "limitation": ["limitation", "risk", "caveat", "weakness", "constraint", "issue"],
                "next_step": ["next step", "future", "improve", "recommend", "validate", "test", "monitor", "extend"],
            },
        },
    ]



def flexible_concept_recall(answer: str, gold_concepts: Dict[str, List[str]]) -> float:
    """Score whether the answer covers flexible concepts rather than exact terms.

    Each concept contains multiple acceptable surface forms. The answer receives
    credit for a concept if any acceptable form appears. This is still lightweight,
    but it is less brittle than requiring one exact keyword.
    """
    if not gold_concepts:
        return 0.0

    low = answer.lower()
    hits = 0
    for variants in gold_concepts.values():
        if any(str(v).lower() in low for v in variants):
            hits += 1
    return hits / len(gold_concepts)


# Backward-compatible alias for older calls.
def simple_keyword_recall(answer: str, gold_keywords: List[str]) -> float:
    if not gold_keywords:
        return 0.0
    return flexible_concept_recall(
        answer,
        {str(keyword): [str(keyword)] for keyword in gold_keywords},
    )


def evaluate_system(rag: TechMPowerRAG) -> List[Dict]:
    eval_questions = make_sample_eval_questions()
    rows = []

    print("=" * 90)
    print("Running project-document evaluation...\n")

    for item in eval_questions:
        result = rag.answer_question(item["question"], mode="qa", role="general")
        recall = flexible_concept_recall(result["answer"], item.get("gold_concepts", {}))
        retrieved_pages = [c.get("page") for c in result.get("citations", []) if c.get("page")]

        row = {
            "question": item["question"],
            "concept_recall": round(recall, 3),
            "top_source": result["citations"][0]["source_file"] if result.get("citations") else None,
            "retrieved_pages": retrieved_pages,
            "num_citations": len(result.get("citations", [])),
        }
        rows.append(row)

        print(f"Q: {item['question']}")
        print(f"Concept recall: {row['concept_recall']}")
        print(f"Top source: {row['top_source']}")
        print(f"Retrieved pages: {row['retrieved_pages']}")
        print("-" * 90)

    avg = np.mean([r["concept_recall"] for r in rows]) if rows else 0.0
    print(f"\nAverage concept recall: {avg:.3f}")
    return rows


