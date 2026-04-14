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

BM25_TOP_K = 8
VECTOR_TOP_K = 8
RERANK_TOP_K = 5

INDEX_DIR = "techmpower_index"
DEFAULT_DOCS_DIR = "."

USE_OPENAI = True
OPENAI_MODEL = "gpt-4o-mini"


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
        "- Do not invent facts beyond the documents."
    )

    return profile_prompt


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
        text = clean_text(text)
        if text:
            pages.append((i + 1, text))
    return pages


def load_docx(filepath: str) -> List[Tuple[Optional[int], str]]:
    doc = DocxDocument(filepath)
    paragraphs = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
    text = "\n".join(paragraphs)
    return [(None, text)] if text else []


def load_document(filepath: str) -> List[Tuple[Optional[int], str]]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return load_pdf(filepath)
    elif ext == ".docx":
        return load_docx(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def build_chunks_for_file(filepath: str) -> List[Chunk]:
    filename = os.path.basename(filepath)
    source_type = detect_source_type(filename)
    raw_units = load_document(filepath)

    output = []
    counter = 0

    for page_num, raw_text in raw_units:
        pieces = split_into_sentential_units(raw_text)
        merged = " ".join(pieces)
        subchunks = chunk_text(merged)

        for sub in subchunks:
            chunk = Chunk(
                chunk_id=f"{filename}_p{page_num or 0}_c{counter}",
                text=sub,
                source_file=filename,
                source_type=source_type,
                page=page_num,
                section=infer_section(sub, source_type),
                aim=infer_aim(sub),
                data_type=infer_data_type(sub),
                sensitivity=infer_sensitivity(sub),
                human_review_required=True
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

        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.reranker = CrossEncoder(self.reranker_model_name)

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
            if fname.lower().endswith((".pdf", ".docx")):
                files.append(os.path.join(docs_dir, fname))
        return sorted(files)

    def build_index(self, docs_dir: str = DEFAULT_DOCS_DIR) -> None:
        files = self._discover_files(docs_dir)
        if not files:
            raise FileNotFoundError(
                f"No PDF/DOCX files found in '{docs_dir}'. "
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
                f"[Context {rank} | score={score:.4f}]\n"
                f"Source: {chunk.source_file} | {loc}\n"
                f"Section: {chunk.section}\n"
                f"Aim: {chunk.aim}\n"
                f"Data Type: {chunk.data_type}\n"
                f"Text: {chunk.text}\n"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

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
                "You are a research assistant for the TechMPower R33 study. "
                "Answer only using the retrieved evidence. "
                "If evidence is insufficient, say so clearly. "
                "Do not make clinical, legal, eligibility, or custody decisions. "
                "Use a concise but informative answer. "
                "End with: 'Human review required.'"
            )
        elif mode == "summary":
            task_prompt = (
                "You are summarizing implementation research evidence for TechMPower. "
                "Use only the retrieved evidence. "
                "Write a concise plain-language summary. "
                "Flag uncertainty clearly. "
                "End with: 'Human review required.'"
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
            "Do not use outside knowledge. "
            "If the retrieved context does not support the answer, say that the evidence is insufficient."
        )

        user_prompt = f"""
Question:
{query}

Retrieved context:
{context}
"""

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.1,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()

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
        user_profile: Optional[Dict] = None
    ) -> Dict:
        if mode not in {"qa", "summary", "coding"}:
            raise ValueError("Mode must be one of: 'qa', 'summary', 'coding'.")

        if role not in {"general", "pm", "engineer", "business"}:
            raise ValueError("Role must be one of: 'general', 'pm', 'engineer', 'business'.")

        if self._blocked_query(query):
            return {
                "mode": mode,
                "role": role,
                "query": query,
                "blocked": True,
                "answer": GUARDRAIL_MESSAGE,
                "citations": [],
                "user_profile": user_profile
            }

        retrieved = self.retrieve(query, top_k=RERANK_TOP_K)
        context = self.format_context(retrieved)

        if USE_OPENAI:
            answer = self._generate_with_openai(
                query=query,
                context=context,
                mode=mode,
                role=role,
                user_profile=user_profile
            )
        else:
            answer = self._heuristic_answer(query, retrieved, mode, role)

        citations = [
            {
                "source_file": chunk.source_file,
                "page": chunk.page,
                "section": chunk.section,
                "aim": chunk.aim,
                "score": round(score, 4)
            }
            for chunk, score in retrieved
        ]

        return {
            "mode": mode,
            "role": role,
            "query": query,
            "blocked": False,
            "answer": answer,
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
        print(f"{i}. {c['source_file']} | {loc} | {c['section']} | {c['aim']} | score={c['score']}")


def list_uploaded_docs(docs_dir: str = ".") -> List[str]:
    files = []
    for fname in sorted(os.listdir(docs_dir)):
        if fname.lower().endswith((".pdf", ".docx")):
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
    return [
        {
            "question": "What is the overall R33 study design?",
            "gold_keywords": ["Hybrid Type II", "stepped wedge", "six county jails", "n=1200"],
        },
        {
            "question": "What kinds of AI uses are allowed in the TechMPower R33 phase?",
            "gold_keywords": ["analytic augmentation", "data quality", "qualitative coding", "cost"],
        },
        {
            "question": "What decisions are prohibited for AI outputs?",
            "gold_keywords": ["clinical", "eligibility", "custody", "parole"],
        },
        {
            "question": "How is human review incorporated in the workflow?",
            "gold_keywords": ["research team", "community member", "human in the loop"],
        },
        {
            "question": "What privacy protections are described for stored data?",
            "gold_keywords": ["de-identified", "encrypted", "password protected", "secure"],
        },
    ]


def simple_keyword_recall(answer: str, gold_keywords: List[str]) -> float:
    if not gold_keywords:
        return 0.0
    low = answer.lower()
    hits = sum(1 for kw in gold_keywords if kw.lower() in low)
    return hits / len(gold_keywords)


def evaluate_system(rag: TechMPowerRAG) -> List[Dict]:
    eval_questions = make_sample_eval_questions()
    rows = []

    print("=" * 90)
    print("Running evaluation...\n")

    for item in eval_questions:
        result = rag.answer_question(item["question"], mode="qa", role="general")
        recall = simple_keyword_recall(result["answer"], item["gold_keywords"])

        row = {
            "question": item["question"],
            "recall": round(recall, 3),
            "top_source": result["citations"][0]["source_file"] if result["citations"] else None
        }
        rows.append(row)

        print(f"Q: {item['question']}")
        print(f"Recall: {row['recall']}")
        print(f"Top source: {row['top_source']}")
        print("-" * 90)

    avg = np.mean([r["recall"] for r in rows]) if rows else 0.0
    print(f"\nAverage keyword recall: {avg:.3f}")
    return rows


# # -*- coding: utf-8 -*-
# """rag_system

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1jQENUbxJTiE2iokdcPeVSeI1-G1D2IJn
# """

# import os
# import re
# import json
# import pickle
# import shutil
# from dataclasses import dataclass
# from typing import List, Dict, Tuple, Optional

# import numpy as np
# from pypdf import PdfReader
# from docx import Document as DocxDocument
# from rank_bm25 import BM25Okapi
# from sentence_transformers import SentenceTransformer, CrossEncoder
# from openai import OpenAI

# try:
#     import faiss
# except ImportError as e:
#     raise ImportError("Please install faiss-cpu: pip install faiss-cpu") from e


# # =========================================================
# # Config
# # =========================================================

# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# CHUNK_SIZE_WORDS = 220
# CHUNK_OVERLAP_WORDS = 50

# BM25_TOP_K = 8
# VECTOR_TOP_K = 8
# RERANK_TOP_K = 5

# INDEX_DIR = "techmpower_index"
# DEFAULT_DOCS_DIR = "."

# USE_OPENAI = True
# OPENAI_MODEL = "gpt-4o-mini"


# # =========================================================
# # Data structure
# # =========================================================

# @dataclass
# class Chunk:
#     chunk_id: str
#     text: str
#     source_file: str
#     source_type: str
#     page: Optional[int]
#     section: str
#     aim: str
#     data_type: str
#     sensitivity: str
#     human_review_required: bool


# # =========================================================
# # Guardrails
# # =========================================================

# BLOCKED_PATTERNS = [
#     r"\beligibility\b",
#     r"\bparole\b",
#     r"\bcustody\b",
#     r"\brisk score\b",
#     r"\brisk prediction\b",
#     r"\bwhich participant\b",
#     r"\bwho should be prioritized\b",
#     r"\bclinical decision\b",
#     r"\bshould this person receive\b",
#     r"\bcriminal behavior\b",
#     r"\bsurveillance\b",
# ]

# GUARDRAIL_MESSAGE = (
#     "This system is limited to analytic augmentation and document-supported research assistance. "
#     "It cannot make participant-level clinical, legal, eligibility, or custody-related decisions. "
#     "Please refer the case for trained human review."
# )

# ROLE_PROMPTS = {
#     "general": (
#         "You are a careful research assistant.\n"
#         "Use ONLY the retrieved evidence.\n\n"
#         "Your task:\n"
#         "- Provide a clear and accurate answer.\n"
#         "- Do not add external knowledge.\n"
#         "- If information is missing, say 'Not specified in the documents.'\n\n"
#         "Output style:\n"
#         "- Concise\n"
#         "- Evidence-grounded\n"
#         "- Neutral tone"
#     ),

#     "pm": (
#         "You are answering as a Product Manager.\n"
#         "Your goal is to translate research evidence into actionable product decisions.\n\n"
#         "Focus on:\n"
#         "- User needs and pain points\n"
#         "- Workflow integration\n"
#         "- Implementation feasibility\n"
#         "- Tradeoffs and risks\n\n"
#         "Output structure:\n"
#         "1. Key insight from evidence\n"
#         "2. Product implication\n"
#         "3. Recommended next step\n\n"
#         "Do NOT focus on technical details unless necessary."
#     ),

#     "engineer": (
#         "You are answering as a Software/ML Engineer.\n"
#         "Your goal is to explain how this would be built or implemented.\n\n"
#         "Focus on:\n"
#         "- System architecture\n"
#         "- Data pipeline\n"
#         "- Model / algorithm choices\n"
#         "- Constraints and risks\n\n"
#         "Output structure:\n"
#         "1. Relevant evidence\n"
#         "2. System design interpretation\n"
#         "3. Implementation approach\n\n"
#         "Be concrete and technical."
#     ),

#     "business": (
#         "You are answering as a Business / Strategy Manager.\n"
#         "Your goal is to evaluate value, scalability, and impact.\n\n"
#         "Focus on:\n"
#         "- Business value\n"
#         "- Cost and ROI\n"
#         "- Scalability and adoption\n"
#         "- Stakeholder impact\n\n"
#         "Output structure:\n"
#         "1. Key evidence\n"
#         "2. Business implication\n"
#         "3. Strategic recommendation\n\n"
#         "Avoid technical jargon."
#     ),
# }

# # =========================================================
# # Utility functions
# # =========================================================

# def ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)


# def clean_text(text: str) -> str:
#     text = text.replace("\x00", " ")
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def tokenize_for_bm25(text: str) -> List[str]:
#     return re.findall(r"[a-zA-Z0-9_/-]+", text.lower())


# def detect_source_type(filename: str) -> str:
#     low = filename.lower()
#     if "workflow" in low:
#         return "Workflow"
#     if "appendix" in low or "ai" in low or "llm" in low:
#         return "AI_Appendix"
#     if "datasheet" in low or "irb" in low or "protocol" in low:
#         return "IRB_Protocol"
#     return "Other"


# def infer_section(text: str, source_type: str) -> str:
#     low = text.lower()
#     if "study design" in low or "hybrid type ii" in low or "stepped wedge" in low:
#         return "Study Design"
#     if "privacy" in low or "data security" in low or "hipaa" in low:
#         return "Privacy & Data Security"
#     if "aim 1" in low or "effectiveness" in low:
#         return "Aim 1"
#     if "aim 2" in low or "implementation" in low or "prism" in low or "re-aim" in low:
#         return "Aim 2"
#     if "aim 3" in low or "cost-effectiveness" in low or "sustainability" in low:
#         return "Aim 3"
#     if "workflow" in low or "human in the loop" in low:
#         return "Workflow"
#     return source_type


# def infer_aim(text: str) -> str:
#     low = text.lower()
#     if "aim 1" in low or "effectiveness" in low:
#         return "Aim 1"
#     if "aim 2" in low or "implementation" in low or "prism" in low or "re-aim" in low:
#         return "Aim 2"
#     if "aim 3" in low or "cost-effectiveness" in low or "sustainability" in low:
#         return "Aim 3"
#     return "Cross-cutting"


# def infer_data_type(text: str) -> str:
#     low = text.lower()
#     if "acasi" in low or "survey" in low:
#         return "Survey"
#     if "interview" in low or "qualitative" in low or "transcript" in low:
#         return "Qualitative Text"
#     if "emr" in low or "medical record" in low:
#         return "Medical Records"
#     if "training" in low or "attendance" in low or "implementation log" in low:
#         return "Training/Implementation"
#     if "cost" in low or "staff time" in low:
#         return "Cost Data"
#     if "workflow" in low:
#         return "Workflow"
#     return "General Research Text"


# def infer_sensitivity(text: str) -> str:
#     low = text.lower()
#     if "phi" in low or "pii" in low or "hipaa" in low or "medical record" in low:
#         return "High"
#     if "de-identified" in low or "aggregate" in low:
#         return "Medium"
#     return "Low"


# def split_into_sentential_units(text: str) -> List[str]:
#     pieces = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z])", text)
#     return [p.strip() for p in pieces if p.strip()]


# def chunk_text(
#     text: str,
#     chunk_size_words: int = CHUNK_SIZE_WORDS,
#     overlap_words: int = CHUNK_OVERLAP_WORDS
# ) -> List[str]:
#     words = text.split()
#     if not words:
#         return []

#     chunks = []
#     start = 0
#     while start < len(words):
#         end = min(len(words), start + chunk_size_words)
#         chunk = " ".join(words[start:end]).strip()
#         if chunk:
#             chunks.append(chunk)
#         if end == len(words):
#             break
#         start = max(0, end - overlap_words)
#     return chunks


# # =========================================================
# # File loading
# # =========================================================

# def load_pdf(filepath: str) -> List[Tuple[int, str]]:
#     reader = PdfReader(filepath)
#     pages = []
#     for i, page in enumerate(reader.pages):
#         try:
#             text = page.extract_text() or ""
#         except Exception:
#             text = ""
#         text = clean_text(text)
#         if text:
#             pages.append((i + 1, text))
#     return pages


# def load_docx(filepath: str) -> List[Tuple[Optional[int], str]]:
#     doc = DocxDocument(filepath)
#     paragraphs = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
#     text = "\n".join(paragraphs)
#     return [(None, text)] if text else []


# def load_document(filepath: str) -> List[Tuple[Optional[int], str]]:
#     ext = os.path.splitext(filepath)[1].lower()
#     if ext == ".pdf":
#         return load_pdf(filepath)
#     elif ext == ".docx":
#         return load_docx(filepath)
#     else:
#         raise ValueError(f"Unsupported file type: {filepath}")


# def build_chunks_for_file(filepath: str) -> List[Chunk]:
#     filename = os.path.basename(filepath)
#     source_type = detect_source_type(filename)
#     raw_units = load_document(filepath)

#     output = []
#     counter = 0

#     for page_num, raw_text in raw_units:
#         pieces = split_into_sentential_units(raw_text)
#         merged = " ".join(pieces)
#         subchunks = chunk_text(merged)

#         for sub in subchunks:
#             chunk = Chunk(
#                 chunk_id=f"{filename}_p{page_num or 0}_c{counter}",
#                 text=sub,
#                 source_file=filename,
#                 source_type=source_type,
#                 page=page_num,
#                 section=infer_section(sub, source_type),
#                 aim=infer_aim(sub),
#                 data_type=infer_data_type(sub),
#                 sensitivity=infer_sensitivity(sub),
#                 human_review_required=True
#             )
#             output.append(chunk)
#             counter += 1

#     return output


# # =========================================================
# # Main RAG class
# # =========================================================

# class TechMPowerRAG:
#     def __init__(
#         self,
#         embed_model_name: str = EMBED_MODEL_NAME,
#         reranker_model_name: str = RERANKER_MODEL_NAME
#     ):
#         self.embed_model_name = embed_model_name
#         self.reranker_model_name = reranker_model_name

#         self.embed_model = SentenceTransformer(self.embed_model_name)
#         self.reranker = CrossEncoder(self.reranker_model_name)

#         self.chunks: List[Chunk] = []
#         self.chunk_texts: List[str] = []
#         self.bm25 = None
#         self.bm25_tokens = None
#         self.embeddings = None
#         self.index = None

#     def _blocked_query(self, query: str) -> bool:
#         q = query.lower()
#         return any(re.search(p, q) for p in BLOCKED_PATTERNS)

#     def _discover_files(self, docs_dir: str) -> List[str]:
#         files = []
#         for fname in os.listdir(docs_dir):
#             if fname.lower().endswith((".pdf", ".docx")):
#                 files.append(os.path.join(docs_dir, fname))
#         return sorted(files)

#     def build_index(self, docs_dir: str = DEFAULT_DOCS_DIR) -> None:
#         files = self._discover_files(docs_dir)
#         if not files:
#             raise FileNotFoundError(
#                 f"No PDF/DOCX files found in '{docs_dir}'. "
#                 "In Colab, your uploaded files are usually in the current directory '.', "
#                 "so use rag.build_index('.')"
#             )

#         all_chunks = []
#         for fp in files:
#             print(f"Processing: {os.path.basename(fp)}")
#             all_chunks.extend(build_chunks_for_file(fp))

#         if not all_chunks:
#             raise ValueError("Files were found, but no text could be extracted.")

#         self.chunks = all_chunks
#         self.chunk_texts = [c.text for c in self.chunks]

#         self.bm25_tokens = [tokenize_for_bm25(t) for t in self.chunk_texts]
#         self.bm25 = BM25Okapi(self.bm25_tokens)

#         print("Encoding dense embeddings...")
#         embeddings = self.embed_model.encode(
#             self.chunk_texts,
#             batch_size=32,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         ).astype("float32")

#         self.embeddings = embeddings

#         dim = embeddings.shape[1]
#         self.index = faiss.IndexFlatIP(dim)
#         self.index.add(embeddings)

#         self.save(INDEX_DIR)
#         print(f"Index built and saved to '{INDEX_DIR}'.")

#     def save(self, out_dir: str = INDEX_DIR) -> None:
#         ensure_dir(out_dir)

#         with open(os.path.join(out_dir, "chunks.pkl"), "wb") as f:
#             pickle.dump(self.chunks, f)

#         with open(os.path.join(out_dir, "bm25_tokens.pkl"), "wb") as f:
#             pickle.dump(self.bm25_tokens, f)

#         np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)
#         faiss.write_index(self.index, os.path.join(out_dir, "faiss.index"))

#     def load(self, out_dir: str = INDEX_DIR) -> None:
#         with open(os.path.join(out_dir, "chunks.pkl"), "rb") as f:
#             self.chunks = pickle.load(f)

#         with open(os.path.join(out_dir, "bm25_tokens.pkl"), "rb") as f:
#             self.bm25_tokens = pickle.load(f)

#         self.chunk_texts = [c.text for c in self.chunks]
#         self.bm25 = BM25Okapi(self.bm25_tokens)
#         self.embeddings = np.load(os.path.join(out_dir, "embeddings.npy")).astype("float32")
#         self.index = faiss.read_index(os.path.join(out_dir, "faiss.index"))

#         print(f"Loaded existing index from '{out_dir}'.")

#     def retrieve(self, query: str, top_k: int = RERANK_TOP_K) -> List[Tuple[Chunk, float]]:
#         if self._blocked_query(query):
#             raise PermissionError(GUARDRAIL_MESSAGE)

#         q_tokens = tokenize_for_bm25(query)
#         bm25_scores = self.bm25.get_scores(q_tokens)
#         bm25_idx = np.argsort(bm25_scores)[::-1][:BM25_TOP_K].tolist()

#         q_emb = self.embed_model.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         ).astype("float32")

#         scores, idxs = self.index.search(q_emb, VECTOR_TOP_K)
#         dense_idx = idxs[0].tolist()

#         candidates = sorted(set(bm25_idx + dense_idx))

#         pairs = [(query, self.chunk_texts[i]) for i in candidates]
#         rerank_scores = self.reranker.predict(pairs)

#         reranked = sorted(
#             [(self.chunks[i], float(score)) for i, score in zip(candidates, rerank_scores)],
#             key=lambda x: x[1],
#             reverse=True
#         )

#         return reranked[:top_k]

#     def format_context(self, retrieved: List[Tuple[Chunk, float]]) -> str:
#         blocks = []
#         for rank, (chunk, score) in enumerate(retrieved, start=1):
#             loc = f"p.{chunk.page}" if chunk.page else "doc"
#             block = (
#                 f"[Context {rank} | score={score:.4f}]\n"
#                 f"Source: {chunk.source_file} | {loc}\n"
#                 f"Section: {chunk.section}\n"
#                 f"Aim: {chunk.aim}\n"
#                 f"Data Type: {chunk.data_type}\n"
#                 f"Text: {chunk.text}\n"
#             )
#             blocks.append(block)
#         return "\n\n".join(blocks)

#     def _generate_with_openai(
#         self,
#         query: str,
#         context: str,
#         mode: str,
#         role: str = "general"
#     ) -> str:
#         api_key = os.environ.get("OPENAI_API_KEY")
#         if not api_key:
#             raise EnvironmentError(
#                 "OPENAI_API_KEY is not set. Please run: "
#                 "os.environ['OPENAI_API_KEY'] = 'your_key'"
#             )

#         client = OpenAI(api_key=api_key)

#         role = (role or "general").lower()
#         role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["general"])

#         if mode == "qa":
#             task_prompt = (
#                 "You are a research assistant for the TechMPower R33 study. "
#                 "Answer only using the retrieved evidence. "
#                 "If evidence is insufficient, say so clearly. "
#                 "Do not make clinical, legal, eligibility, or custody decisions. "
#                 "Use a concise but informative answer. "
#                 "End with: 'Human review required.'"
#             )
#         elif mode == "summary":
#             task_prompt = (
#                 "You are summarizing implementation research evidence for TechMPower. "
#                 "Use only the retrieved evidence. "
#                 "Write a concise plain-language summary. "
#                 "Flag uncertainty clearly. "
#                 "End with: 'Human review required.'"
#             )
#         elif mode == "coding":
#             task_prompt = (
#                 "You are assisting with first-pass qualitative coding for implementation research. "
#                 "Given the evidence, suggest 1-3 candidate codes and possible PRISM/RE-AIM mapping. "
#                 "Do not claim final coding certainty. "
#                 "End with: 'Final coding requires human review.'"
#             )
#         else:
#             task_prompt = "Use only the retrieved evidence."

#         system_prompt = (
#             f"{role_prompt}\n\n"
#             f"{task_prompt}\n\n"
#             "Do not use outside knowledge. "
#             "If the retrieved context does not support the answer, say that the evidence is insufficient."
#         )

#         user_prompt = f"""
# Question:
# {query}

# Retrieved context:
# {context}
# """

#         response = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             temperature=0.1,
#             max_tokens=300,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ]
#         )

#         return response.choices[0].message.content.strip()

#     def _heuristic_answer(
#         self,
#         query: str,
#         retrieved: List[Tuple[Chunk, float]],
#         mode: str,
#         role: str = "general"
#     ) -> str:
#         combined = " ".join(chunk.text for chunk, _ in retrieved)
#         low = combined.lower()

#         role = (role or "general").lower()
#         role_prefix = {
#             "pm": "Product perspective",
#             "engineer": "Engineering perspective",
#             "business": "Business perspective",
#             "general": "General research perspective"
#         }.get(role, "General research perspective")

#         if mode == "summary":
#             return (
#                 f"{role_prefix} summary:\n"
#                 f"{combined[:1200]}...\n\n"
#                 "Human review required before using this output in reporting."
#             )

#         if mode == "coding":
#             codes = []
#             if "stigma" in low:
#                 codes.append("Stigma / negative attitudes")
#             if "training" in low:
#                 codes.append("Training / workforce preparation")
#             if "coordination" in low or "linkage" in low:
#                 codes.append("Care coordination / linkage")
#             if "cost" in low:
#                 codes.append("Cost / sustainability")
#             if "technology" in low or "telehealth" in low:
#                 codes.append("Technology-enabled implementation")
#             if "fidelity" in low:
#                 codes.append("Implementation fidelity")

#             if not codes:
#                 codes = ["Implementation process", "Contextual barrier", "Needs human review"]

#             return (
#                 f"{role_prefix} first-pass coding suggestion:\n"
#                 f"- Candidate codes: {', '.join(codes[:5])}\n"
#                 "- These are preliminary analytic suggestions only.\n"
#                 "- Final coding requires human review."
#             )

#         return (
#             f"{role_prefix} answer based on retrieved study documents:\n"
#             f"{combined[:1200]}...\n\n"
#             "This answer is grounded in the retrieved protocol/workflow materials. "
#             "Human review is required for final interpretation."
#         )

#     def answer_question(self, query: str, mode: str = "qa", role: str = "general") -> Dict:
#         if mode not in {"qa", "summary", "coding"}:
#             raise ValueError("Mode must be one of: 'qa', 'summary', 'coding'.")

#         if role not in {"general", "pm", "engineer", "business"}:
#             raise ValueError("Role must be one of: 'general', 'pm', 'engineer', 'business'.")

#         if self._blocked_query(query):
#             return {
#                 "mode": mode,
#                 "role": role,
#                 "query": query,
#                 "blocked": True,
#                 "answer": GUARDRAIL_MESSAGE,
#                 "citations": []
#             }

#         retrieved = self.retrieve(query, top_k=RERANK_TOP_K)
#         context = self.format_context(retrieved)

#         if USE_OPENAI:
#             answer = self._generate_with_openai(query, context, mode, role)
#         else:
#             answer = self._heuristic_answer(query, retrieved, mode, role)

#         citations = [
#             {
#                 "source_file": chunk.source_file,
#                 "page": chunk.page,
#                 "section": chunk.section,
#                 "aim": chunk.aim,
#                 "score": round(score, 4)
#             }
#             for chunk, score in retrieved
#         ]

#         return {
#             "mode": mode,
#             "role": role,
#             "query": query,
#             "blocked": False,
#             "answer": answer,
#             "citations": citations,
#             "retrieved_context": context
#         }


# # =========================================================
# # Convenience helpers for Colab
# # =========================================================

# def print_answer(result: Dict) -> None:
#     print("=" * 90)
#     print("MODE:", result["mode"])
#     print("ROLE:", result.get("role", "general"))
#     print("QUERY:", result["query"])
#     print("\nANSWER:\n")
#     print(result["answer"])

#     print("\nTOP CITATIONS:")
#     if not result["citations"]:
#         print("No citations.")
#         return

#     for i, c in enumerate(result["citations"], start=1):
#         loc = f"p.{c['page']}" if c["page"] else "doc"
#         print(f"{i}. {c['source_file']} | {loc} | {c['section']} | {c['aim']} | score={c['score']}")


# def list_uploaded_docs(docs_dir: str = ".") -> List[str]:
#     files = []
#     for fname in sorted(os.listdir(docs_dir)):
#         if fname.lower().endswith((".pdf", ".docx")):
#             files.append(fname)
#     return files


# def move_docs_to_folder(file_list: List[str], target_dir: str = "docs") -> None:
#     ensure_dir(target_dir)
#     for f in file_list:
#         if os.path.exists(f):
#             shutil.move(f, os.path.join(target_dir, f))
#             print(f"Moved: {f} -> {target_dir}/{f}")
#         else:
#             print(f"Not found: {f}")


# def initialize_rag(docs_dir: str = ".", force_rebuild: bool = False) -> TechMPowerRAG:
#     rag = TechMPowerRAG()

#     index_path = os.path.join(INDEX_DIR, "faiss.index")
#     if os.path.exists(index_path) and not force_rebuild:
#         rag.load(INDEX_DIR)
#     else:
#         rag.build_index(docs_dir)

#     return rag


# # =========================================================
# # Evaluation
# # =========================================================

# def make_sample_eval_questions() -> List[Dict]:
#     return [
#         {
#             "question": "What is the overall R33 study design?",
#             "gold_keywords": ["Hybrid Type II", "stepped wedge", "six county jails", "n=1200"],
#         },
#         {
#             "question": "What kinds of AI uses are allowed in the TechMPower R33 phase?",
#             "gold_keywords": ["analytic augmentation", "data quality", "qualitative coding", "cost"],
#         },
#         {
#             "question": "What decisions are prohibited for AI outputs?",
#             "gold_keywords": ["clinical", "eligibility", "custody", "parole"],
#         },
#         {
#             "question": "How is human review incorporated in the workflow?",
#             "gold_keywords": ["research team", "community member", "human in the loop"],
#         },
#         {
#             "question": "What privacy protections are described for stored data?",
#             "gold_keywords": ["de-identified", "encrypted", "password protected", "secure"],
#         },
#     ]


# def simple_keyword_recall(answer: str, gold_keywords: List[str]) -> float:
#     if not gold_keywords:
#         return 0.0
#     low = answer.lower()
#     hits = sum(1 for kw in gold_keywords if kw.lower() in low)
#     return hits / len(gold_keywords)


# def evaluate_system(rag: TechMPowerRAG) -> List[Dict]:
#     eval_questions = make_sample_eval_questions()
#     rows = []

#     print("=" * 90)
#     print("Running evaluation...\n")

#     for item in eval_questions:
#         result = rag.answer_question(item["question"], mode="qa", role="general")
#         recall = simple_keyword_recall(result["answer"], item["gold_keywords"])

#         row = {
#             "question": item["question"],
#             "recall": round(recall, 3),
#             "top_source": result["citations"][0]["source_file"] if result["citations"] else None
#         }
#         rows.append(row)

#         print(f"Q: {item['question']}")
#         print(f"Recall: {row['recall']}")
#         print(f"Top source: {row['top_source']}")
#         print("-" * 90)

#     avg = np.mean([r["recall"] for r in rows]) if rows else 0.0
#     print(f"\nAverage keyword recall: {avg:.3f}")
#     return rows
