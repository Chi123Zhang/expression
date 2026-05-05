

"""
Expression Layer for the Background-Aware Expression Agent.

This module is responsible for two things:
1. Building an explicit ExpressionPlan from the user's background and query context.
2. Rewriting a neutral/base explanation into a personalized explanation.

The goal is to make personalization visible and controllable instead of hiding it
inside one large prompt.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass
class ExpressionPlan:
    """A structured plan describing how the final answer should be expressed."""

    target_audience: str = "general"
    explanation_goal: str = "general_understanding"
    detail_level: str = "medium"
    tone: str = "clear and helpful"
    structure_style: str = "structured"
    use_analogy: bool = False
    jargon_policy: str = "define_if_used"
    emphasis: List[str] = field(default_factory=list)
    de_emphasis: List[str] = field(default_factory=list)
    include_business_impact: bool = False
    include_example: bool = True
    example_style: str = "simple example"
    final_format: str = "short structured explanation"
    grounding_policy: str = "do_not_invent_unsupported_specifics"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExpressionQualityReport:
    """Explainable quality checks for the final personalized answer.

    This is intentionally lightweight and deterministic. It helps the project
    look less like a prompt wrapper and more like an inspectable agent system.
    """

    audience_alignment_score: float = 0.0
    structure_compliance_score: float = 0.0
    specificity_score: float = 0.0
    grounding_risk_score: float = 0.0
    retrieval_generation_separation_score: float = 0.0
    risk_flags: List[str] = field(default_factory=list)
    missing_expected_sections: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


QUERY_TO_GOAL: Dict[str, str] = {
    "concept_explanation": "conceptual_understanding",
    "project_explanation": "system_understanding",
    "comparison_question": "decision_support",
    "workflow_explanation": "process_understanding",
    "document_based_question": "document_grounded_explanation",
    "clarification_needed": "clarify_user_need_before_answering",
}


ROLE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "pm": {
        "target_audience": "product_manager",
        "detail_level": "medium",
        "tone": "clear, practical, structured, and delivery-oriented",
        "structure_style": "product_workflow_layered",
        "use_analogy": True,
        "jargon_policy": "minimize_and_define_if_used",
        "emphasis": [
            "user problem and product goal",
            "end-to-end workflow",
            "module responsibilities",
            "user journey",
            "dependencies and handoffs",
            "risks and blockers",
            "success metrics",
            "delivery implications",
            "why it matters for product execution",
        ],
        "de_emphasis": [
            "low-level implementation details",
            "infrastructure internals",
            "algorithmic details unless necessary",
            "code-level details",
        ],
        "include_business_impact": True,
        "include_example": True,
        "example_style": "product workflow example with user scenario and stakeholder handoff",
        "final_format": "product-facing breakdown with summary, workflow, risks, and PM takeaway",
        "grounding_policy": "do_not_invent_requirements_metrics_or_roadmap_details",
    },
    "product_manager": {
        "target_audience": "product_manager",
        "detail_level": "medium",
        "tone": "clear, practical, structured, and delivery-oriented",
        "structure_style": "product_workflow_layered",
        "use_analogy": True,
        "jargon_policy": "minimize_and_define_if_used",
        "emphasis": [
            "user problem and product goal",
            "end-to-end workflow",
            "module responsibilities",
            "user journey",
            "dependencies and handoffs",
            "risks and blockers",
            "success metrics",
            "delivery implications",
            "why it matters for product execution",
        ],
        "de_emphasis": [
            "low-level implementation details",
            "infrastructure internals",
            "algorithmic details unless necessary",
            "code-level details",
        ],
        "include_business_impact": True,
        "include_example": True,
        "example_style": "product workflow example with user scenario and stakeholder handoff",
        "final_format": "product-facing breakdown with summary, workflow, risks, and PM takeaway",
        "grounding_policy": "do_not_invent_requirements_metrics_or_roadmap_details",
    },
    "engineer": {
        "target_audience": "engineer",
        "detail_level": "high",
        "tone": "structured, precise, and implementation-oriented",
        "structure_style": "implementation_layered",
        "use_analogy": False,
        "jargon_policy": "allow_technical_terms_with_context",
        "emphasis": [
            "system architecture",
            "module responsibilities",
            "data flow",
            "control flow",
            "interfaces and inputs/outputs",
            "implementation steps",
            "tradeoffs",
            "failure modes",
        ],
        "de_emphasis": [
            "generic high-level explanation",
            "business-only framing",
            "over-simplified analogies",
        ],
        "include_business_impact": False,
        "include_example": True,
        "example_style": "implementation-oriented example with modules and data flow",
        "final_format": "structured implementation breakdown with sections and bullet points",
        "grounding_policy": "do_not_invent_unsupported_specifics",
    },
    "business": {
        "target_audience": "business_owner",
        "detail_level": "low_to_medium",
        "tone": "executive, concise, outcome-focused, and decision-oriented",
        "structure_style": "executive_decision_summary",
        "use_analogy": True,
        "jargon_policy": "avoid_jargon",
        "emphasis": [
            "business problem",
            "business value",
            "expected outcome",
            "customer or operational impact",
            "cost, risk, and efficiency implications",
            "decision points",
            "timeline or implementation effort at a high level",
            "what leadership needs to know",
        ],
        "de_emphasis": [
            "technical implementation details",
            "model internals",
            "code-level details",
            "deep architecture unless required for a decision",
        ],
        "include_business_impact": True,
        "include_example": False,
        "example_style": "business outcome example only if grounded in the provided context",
        "final_format": "executive summary with value, risk, decision points, and next step",
        "grounding_policy": "do_not_invent_roi_cost_timeline_or_customer_impact",
    },
    "business_owner": {
        "target_audience": "business_owner",
        "detail_level": "low_to_medium",
        "tone": "executive, concise, outcome-focused, and decision-oriented",
        "structure_style": "executive_decision_summary",
        "use_analogy": True,
        "jargon_policy": "avoid_jargon",
        "emphasis": [
            "business problem",
            "business value",
            "expected outcome",
            "customer or operational impact",
            "cost, risk, and efficiency implications",
            "decision points",
            "timeline or implementation effort at a high level",
            "what leadership needs to know",
        ],
        "de_emphasis": [
            "technical implementation details",
            "model internals",
            "code-level details",
            "deep architecture unless required for a decision",
        ],
        "include_business_impact": True,
        "include_example": False,
        "example_style": "business outcome example only if grounded in the provided context",
        "final_format": "executive summary with value, risk, decision points, and next step",
        "grounding_policy": "do_not_invent_roi_cost_timeline_or_customer_impact",
    },
    "general": {
        "target_audience": "general",
        "detail_level": "medium",
        "tone": "clear and helpful",
        "structure_style": "structured",
        "use_analogy": False,
        "jargon_policy": "define_if_used",
        "emphasis": ["core idea", "why it matters", "clear structure", "simple explanation"],
        "de_emphasis": ["unnecessary complexity"],
        "include_business_impact": False,
        "include_example": True,
        "example_style": "simple example",
        "final_format": "short structured explanation",
        "grounding_policy": "do_not_invent_unsupported_specifics",
    },
}


QUERY_ADJUSTMENTS: Dict[str, Dict[str, Any]] = {
    "concept_explanation": {
        "include_example": True,
    },
    "project_explanation": {
        "structure_style": "artifact_aware_project_analysis",
        "include_example": False,
        "emphasis": [
            "artifact type and scope",
            "project objective",
            "end-to-end workflow",
            "data sources or inputs",
            "system, model, or module roles",
            "evaluation results and metrics",
            "technical risks and limitations",
            "concrete next steps",
        ],
    },
    "comparison_question": {
        "structure_style": "compare_tradeoffs",
        "include_example": False,
        "emphasis": [
            "decision criteria",
            "tradeoffs",
            "risks",
            "implementation effort",
            "business or product impact",
            "recommendation under constraints",
        ],
    },
    "workflow_explanation": {
        "structure_style": "step_by_step",
        "emphasis": [
            "sequence",
            "dependencies",
            "handoffs",
            "inputs and outputs at each step",
            "why each step exists",
        ],
    },
    "document_based_question": {
        "structure_style": "grounded_summary_then_explanation",
        "include_example": False,
    },
}
ROLE_OUTPUT_CONTRACTS: Dict[str, str] = {
    "engineer": """
Use exactly this structure for engineer-facing answers:
1. Technical summary: one sentence.
2. Components: list the main modules and each module's responsibility.
3. Retrieval / data-control flow: explain the concrete sequence from user query to final answer.
4. Interfaces: describe what each component receives and returns when relevant.
5. Implementation notes: give concrete implementation details, design alternatives, chunking/retrieval/reranking decisions, evaluation hooks, and failure modes. When relevant, separate retrieval evaluation from generation/answer-quality evaluation.
6. Useful implementation references: include 2-4 concise references by name, such as FAISS, Chroma, BM25, cross-encoder reranking, LangChain, LlamaIndex, or OpenAI embeddings, only when relevant to the topic.
Do not include executive/business sections. Do not add ROI, leadership next steps, or business-value framing unless the query explicitly asks for it.
Avoid generic statements like "choose efficient retrieval algorithms" unless you also name concrete methods, tradeoffs, evaluation criteria, or implementation options.
Do not invent exact model dimensions, chunk sizes, index types, costs, latency numbers, or library choices unless they are explicitly provided by the base explanation, retrieved background, or project config. When a value is not grounded, present it as a tunable option or practical starting range instead of a fixed fact.
""".strip(),
    "product_manager": """
Use exactly this structure for PM-facing answers:
1. Product summary: one or two sentences.
2. User problem / product goal: explain what user need or workflow problem this solves.
3. Workflow: describe the end-to-end flow in simple steps.
4. Dependencies and handoffs: call out what modules, teams, or inputs the feature depends on.
5. Risks / open questions: list practical product risks or unclear requirements.
6. Success metrics / PM takeaway: end with how a PM would evaluate or move this forward.
Avoid code-level detail and deep architecture. Do not write like an engineer or executive.
""".strip(),
    "business_owner": """
Use exactly this structure for business-owner-facing answers:
1. Business takeaway: start with the decision-relevant bottom line.
2. Value / expected outcome: explain what business value this creates.
3. Customer or operational impact: explain who benefits and how.
4. Risks and decision points: explain what leadership should evaluate.
5. Effort / timeline at a high level: only mention high-level effort if supported; do not invent exact timelines or costs.
6. Recommended next step: end with a practical leadership next step.
Avoid technical jargon, module-level implementation, code details, and engineer-style architecture sections.
""".strip(),
    "general": """
Use a clear structure:
1. Short definition.
2. How it works.
3. Why it matters.
4. Simple example only if useful.
""".strip(),
}


# Project document output contracts and expected sections
PROJECT_DOCUMENT_OUTPUT_CONTRACTS: Dict[str, str] = {
    "engineer": """
Write an engineer-facing project document answer. Do NOT force every project into the same fixed template.

First infer the artifact type from the base explanation and retrieved evidence:
- modeling / ML / statistics report
- software / web app / README
- dashboard / BI project
- data pipeline / ETL project
- RAG / AI-agent system
- research proposal / methodology document
- mixed project portfolio

Choose section headings that fit the artifact. For complex engineer-facing project answers, use 6-9 sections selected from the list below, not necessarily all of them:
- Project overview
- Artifact type and scope
- System architecture
- Workflow architecture
- Data sources and target
- Feature engineering
- Data preparation / preprocessing
- Modeling approach
- Model / module roles
- Evaluation results
- Metrics and interpretation
- Implementation details
- Technical risks and limitations
- Engineering takeaway / next steps

Depth rules:
1. Prefer concrete workflow and implementation details over generic summaries.
2. If the document contains a pipeline, reconstruct the pipeline step by step from inputs to outputs.
3. If the document contains multiple workflows, separate them clearly, for example causal-impact workflow versus forecasting workflow, retrieval workflow versus generation workflow, or frontend workflow versus backend workflow.
4. If the document contains numeric metrics, include the actual metrics and briefly interpret whether they imply strong performance, weak performance, overfitting, underfitting, noisy signal, or validation gaps.
5. If different models/modules serve different roles, separate them clearly instead of calling one overall model "selected."
6. If a model is labeled "best" in the document but has weak metrics, explicitly distinguish documented label from empirical performance.
7. If the artifact is a README, website, dashboard, app, or software project, focus on tech stack, architecture, file structure, UI behavior, deployment, customization workflow, and engineering risks. Do not force dataset/model sections unless the document actually contains them.
8. If the artifact is a modeling report, focus on dataset, target, preprocessing, models, metrics, limitations, validation, and reproducibility.
9. If the artifact is a research/methodology proposal, focus on research goal, workflow, assumptions, evaluation challenges, risks, and next implementation steps.
10. End with concrete engineering next steps such as reproduction, validation, monitoring, debugging, data enrichment, model tuning, or pipeline separation.
11. Do not invent software modules, metrics, sample sizes, model names, business outcomes, or selected-model claims that are not supported by the base explanation or retrieved evidence.
""".strip(),
    "product_manager": """
Write a PM-facing project document answer. Do NOT force every project into the same fixed template.

First infer the artifact type from the base explanation and retrieved evidence:
- modeling / ML / statistics report
- software / web app / README
- dashboard / BI project
- data pipeline / ETL project
- RAG / AI-agent system
- research proposal / methodology document
- mixed project portfolio

Choose section headings that fit the artifact. For most PM-facing project answers, use 5-7 sections selected from the list below, not necessarily all of them:
- Project overview
- User problem / product goal
- Users or stakeholders
- Workflow / user journey
- Core capabilities
- Inputs and outputs
- Key result or current status
- Risks / open questions
- Success criteria
- PM takeaway and next step

Rules:
1. Translate the project into the user need, decision workflow, or operational problem it supports.
2. Explain the workflow in practical terms without over-focusing on code internals.
3. For ML/modeling projects, explain what decision the model supports, what the output means, and what validation risk remains.
4. For software, dashboard, website, or app projects, explain the user-facing functionality, dependencies, deployment/readiness, and adoption risks.
5. Distinguish documented results from potential value. Do not turn a possible use case into a proven outcome.
6. Do not invent requirements, metrics, roadmap items, user impact, stakeholder claims, adoption, or deployment status unless supported by the base explanation or retrieved evidence.
""".strip(),
    "business_owner": """
Write a business-owner-facing project document answer. Do NOT force every project into the same fixed template.

First infer the artifact type from the base explanation and retrieved evidence:
- modeling / ML / statistics report
- software / web app / README
- dashboard / BI project
- data pipeline / ETL project
- RAG / AI-agent system
- research proposal / methodology document
- mixed project portfolio

Choose section headings that fit the artifact. For most business-facing project answers, use 4-6 sections selected from the list below, not necessarily all of them:
- Business takeaway
- What the project does
- Problem / opportunity
- Value or decision support
- Documented results
- Potential value, if validated
- Risks and limitations
- Recommended next step

Rules:
1. Start with the decision-relevant bottom line.
2. Explain the project in plain language while preserving the real evidence.
3. Clearly separate documented outcomes from potential value.
4. For modeling projects, explain whether the evidence supports reliable decision-making or whether it remains exploratory.
5. For software/dashboard/website projects, explain what capability it provides and what would be needed for adoption or production use.
6. Avoid technical module/interface language unless it is necessary for a leadership decision.
7. Do not invent ROI, revenue impact, cost savings, timelines, customer impact, operational benefits, deployment status, or business success unless supported by the base explanation or retrieved evidence.
""".strip(),
    "general": """
Write a clear general-audience project document answer. Do NOT force every project into the same fixed template.

First infer what kind of artifact this is: modeling report, software README, dashboard, data pipeline, RAG/AI-agent system, research proposal, or mixed portfolio.

Choose section headings that fit the artifact. For most general project answers, use 4-6 sections selected from the list below, not necessarily all of them:
- What the project is
- What problem it tries to solve
- What data / inputs it uses
- How it works
- What methods / tools it uses
- What results it reports
- What limitations it has
- What could be improved next

Rules:
1. Explain the real project, not just the title.
2. Use concrete details from the base explanation and retrieved evidence.
3. If the document does not provide a detail, say it is not specified in the retrieved evidence.
4. Do not invent unsupported metrics, model names, business outcomes, deployment claims, or famous-case background.
""".strip(),
}


PROJECT_DOCUMENT_EXPECTED_SECTIONS: Dict[str, List[str]] = {
    "engineer": [
        "Project overview",
        "Data sources",
        "Modeling",
        "Evaluation",
        "Technical risks",
        "Engineering takeaway",
    ],
    "product_manager": [
        "Project overview",
        "Workflow",
        "Risks",
        "PM takeaway",
    ],
    "business_owner": [
        "Business takeaway",
        "What the project does",
        "Risks",
        "Recommended next step",
    ],
    "general": [
        "What the project is",
        "How it works",
        "What limitations it has",
    ],
}

# Generic expected sections and specificity keywords for non-project answers
PROJECT_DOCUMENT_SPECIFICITY_KEYWORDS: Dict[str, List[str]] = {
    "engineer": [
        "workflow",
        "pipeline",
        "data",
        "target",
        "feature",
        "model",
        "metric",
        "r²",
        "rmse",
        "mae",
        "aic",
        "bic",
        "validation",
        "risk",
        "limitation",
        "next step",
    ],
    "product_manager": [
        "user",
        "workflow",
        "goal",
        "output",
        "risk",
        "validation",
        "next step",
    ],
    "business_owner": [
        "value",
        "decision",
        "documented",
        "potential",
        "risk",
        "limitation",
        "next step",
    ],
    "general": [
        "project",
        "goal",
        "data",
        "method",
        "result",
        "limitation",
    ],
}


# Expected sections and specificity keywords for each role/audience


# Expected sections and specificity keywords for each role/audience
ROLE_EXPECTED_SECTIONS: Dict[str, List[str]] = {
    "engineer": [
        "Technical summary",
        "Components",
        "Retrieval / data-control flow",
        "Interfaces",
        "Implementation notes",
        "Useful implementation references",
    ],
    "product_manager": [
        "Product summary",
        "User problem / product goal",
        "Workflow",
        "Dependencies and handoffs",
        "Risks / open questions",
        "Success metrics / PM takeaway",
    ],
    "business_owner": [
        "Business takeaway",
        "Value / expected outcome",
        "Customer or operational impact",
        "Risks and decision points",
        "Effort / timeline at a high level",
        "Recommended next step",
    ],
    "general": [
        "Short definition",
        "How it works",
        "Why it matters",
    ],
}

ROLE_SPECIFICITY_KEYWORDS: Dict[str, List[str]] = {
    "engineer": [
        "api",
        "input",
        "output",
        "interface",
        "module",
        "retrieval",
        "embedding",
        "index",
        "rerank",
        "latency",
        "evaluation",
        "failure mode",
    ],
    "product_manager": [
        "user",
        "workflow",
        "requirement",
        "dependency",
        "handoff",
        "risk",
        "metric",
        "success",
    ],
    "business_owner": [
        "value",
        "outcome",
        "risk",
        "decision",
        "customer",
        "operation",
        "cost",
        "effort",
    ],
    "general": [
        "idea",
        "works",
        "matters",
        "example",
    ],
}




def _is_project_document_question(query_understanding: Optional[Dict[str, Any]] = None) -> bool:
    query_understanding = query_understanding or {}
    query_text = " ".join(
        str(query_understanding.get(key, ""))
        for key in ["query_type", "topic", "intent", "domain"]
    ).lower()

    project_markers = [
        "project",
        "uploaded project",
        "project_explanation",
        "document_based_question",
        "summarize_evidence",
        "project documents",
        "uploaded document",
        "portfolio document",
        "case study",
        "report",
    ]
    return any(marker in query_text for marker in project_markers)


def _get_role_output_contract(
    expression_plan: Dict[str, Any],
    query_understanding: Optional[Dict[str, Any]] = None,
) -> str:
    audience = expression_plan.get("target_audience", "general")

    if _is_project_document_question(query_understanding):
        return PROJECT_DOCUMENT_OUTPUT_CONTRACTS.get(
            audience,
            PROJECT_DOCUMENT_OUTPUT_CONTRACTS["general"],
        )

    return ROLE_OUTPUT_CONTRACTS.get(audience, ROLE_OUTPUT_CONTRACTS["general"])



# Quality evaluation helpers
def _get_expected_sections(
    expression_plan: Dict[str, Any],
    query_understanding: Optional[Dict[str, Any]] = None,
) -> List[str]:
    audience = expression_plan.get("target_audience", "general")

    if _is_project_document_question(query_understanding):
        return PROJECT_DOCUMENT_EXPECTED_SECTIONS.get(
            audience,
            PROJECT_DOCUMENT_EXPECTED_SECTIONS["general"],
        )

    return ROLE_EXPECTED_SECTIONS.get(audience, ROLE_EXPECTED_SECTIONS["general"])


def _score_keyword_specificity(text: str, keywords: List[str]) -> float:
    if not text or not keywords:
        return 0.0

    normalized_text = text.lower()
    matched = sum(1 for keyword in keywords if keyword.lower() in normalized_text)
    return round(min(1.0, matched / max(1, len(keywords) * 0.6)), 2)


def _score_section_compliance(text: str, expected_sections: List[str]) -> Tuple[float, List[str]]:
    if not text or not expected_sections:
        return 0.0, expected_sections

    normalized_text = text.lower()
    missing_sections = [
        section
        for section in expected_sections
        if section.lower() not in normalized_text
    ]
    score = 1.0 - (len(missing_sections) / max(1, len(expected_sections)))
    return round(score, 2), missing_sections


def evaluate_expression_quality(
    final_explanation: str,
    expression_plan: Dict[str, Any],
    query_understanding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate whether the final answer followed the expression plan.

    This is not a factual correctness evaluator. It checks whether the expression
    layer actually expressed the answer in the intended audience format.
    """

    audience = expression_plan.get("target_audience", "general")
    expected_sections = _get_expected_sections(expression_plan, query_understanding=query_understanding)
    section_score, missing_sections = _score_section_compliance(
        final_explanation,
        expected_sections,
    )

    if _is_project_document_question(query_understanding):
        specificity_keywords = PROJECT_DOCUMENT_SPECIFICITY_KEYWORDS.get(
            audience,
            PROJECT_DOCUMENT_SPECIFICITY_KEYWORDS["general"],
        )
    else:
        specificity_keywords = ROLE_SPECIFICITY_KEYWORDS.get(
            audience,
            ROLE_SPECIFICITY_KEYWORDS["general"],
        )
    specificity_score = _score_keyword_specificity(final_explanation, specificity_keywords)

    risk_flags: List[str] = []
    suggestions: List[str] = []

    generic_phrases = [
        "choose efficient retrieval algorithms",
        "ensure data quality",
        "improve accuracy",
        "optimize performance",
        "make it scalable",
    ]

    unsupported_specific_patterns = [
        "dimension of 768",
        "dimension of 1536",
        "dimension of 3072",
        "100-200 tokens",
        "100 to 200 tokens",
        "use faiss with hnsw",
        "choose faiss with hnsw",
        "must use hnsw",
        "must use openai embeddings",
        "guaranteed accuracy",
        "guaranteed performance",
    ]

    retrieval_terms = ["recall@k", "mrr", "ndcg", "coverage", "retrieval quality", "latency"]
    answer_quality_terms = ["groundedness", "faithfulness", "citation correctness", "answer quality", "human review"]

    normalized_text = final_explanation.lower()
    for phrase in generic_phrases:
        if phrase in normalized_text:
            risk_flags.append(f"Generic phrase detected: '{phrase}'")
            suggestions.append("Replace generic implementation wording with concrete methods, parameters, or tradeoffs.")

    unsupported_specific_count = 0
    for pattern in unsupported_specific_patterns:
        if pattern in normalized_text:
            unsupported_specific_count += 1
            risk_flags.append(f"Potentially unsupported fixed implementation detail detected: '{pattern}'")
            suggestions.append("Avoid hard-coded implementation defaults unless they come from retrieved context or config; present them as options, ranges, or tradeoffs instead.")

    grounding_risk_score = round(min(1.0, unsupported_specific_count / 3), 2)

    retrieval_generation_separation_score = 0.0
    topic_text = " ".join(
        str((query_understanding or {}).get(key, ""))
        for key in ["topic", "domain", "intent", "query_type"]
    ).lower()
    is_rag_related = any(term in topic_text or term in normalized_text for term in ["rag", "retrieval", "embedding", "vector"])
    if is_rag_related and audience == "engineer":
        retrieval_hits = sum(1 for term in retrieval_terms if term in normalized_text)
        answer_quality_hits = sum(1 for term in answer_quality_terms if term in normalized_text)
        retrieval_generation_separation_score = round(
            min(1.0, (min(retrieval_hits, 3) / 3) * 0.5 + (min(answer_quality_hits, 3) / 3) * 0.5),
            2,
        )
        if retrieval_generation_separation_score < 0.5:
            risk_flags.append("RAG answer does not clearly separate retrieval evaluation from final answer evaluation.")
            suggestions.append("For RAG answers, explicitly evaluate retrieval quality separately from groundedness, faithfulness, citation correctness, and usefulness of the final answer.")

    if audience == "engineer" and specificity_score < 0.65:
        risk_flags.append("Engineer answer may be too generic.")
        suggestions.append("Add concrete implementation details such as APIs, indexes, chunking, reranking, metrics, or failure modes.")

    if audience in {"product_manager", "business_owner"} and "code" in normalized_text:
        risk_flags.append("Non-engineer answer may contain code-level detail.")
        suggestions.append("Move implementation detail into a dependency or risk framing instead of code-level explanation.")

    if missing_sections:
        suggestions.append("Add the missing audience-specific sections so the output contract is visibly satisfied.")

    audience_alignment_score = round((section_score * 0.6) + (specificity_score * 0.4), 2)

    report = ExpressionQualityReport(
        audience_alignment_score=audience_alignment_score,
        structure_compliance_score=section_score,
        specificity_score=specificity_score,
        grounding_risk_score=grounding_risk_score,
        retrieval_generation_separation_score=retrieval_generation_separation_score,
        risk_flags=list(dict.fromkeys(risk_flags)),
        missing_expected_sections=missing_sections,
        improvement_suggestions=list(dict.fromkeys(suggestions)),
    )
    return report.to_dict()


# Topic-specific engineer guidance helper
def _get_topic_specific_engineer_guidance(query_understanding: Optional[Dict[str, Any]]) -> str:
    if not query_understanding:
        return ""

    topic_text = " ".join(
        str(query_understanding.get(key, ""))
        for key in ["topic", "domain", "intent", "query_type"]
    ).lower()

    if any(term in topic_text for term in ["rag", "retrieval", "retrieval-augmented", "vector", "embedding"]):
        return """
For RAG/retrieval questions, make the engineer-facing answer concrete and implementation-grade.

Cover the retrieval pipeline in this order when relevant:
1. Ingestion: file loader, OCR/text extraction if needed, cleaning, metadata normalization.
2. Chunking: chunk size as a tunable range, overlap, semantic vs fixed-size chunking, parent-child chunks, metadata schema. Do not claim a single exact chunk size unless it is provided by config or retrieved context.
3. Embeddings: model choice, embedding dimension inferred from the selected model, batching, versioning, refresh strategy. Do not invent a dimension such as 768/1536/3072 unless the model/config explicitly supports it.
4. Index/storage: FAISS, Chroma, Pinecone, Weaviate, pgvector; HNSW vs IVF vs Flat as design alternatives; persistence and metadata filters. Explain tradeoffs instead of asserting one index type as the default.
5. Retrieval: dense top-k, BM25, hybrid retrieval, score fusion, recency or access-control filters.
6. Reranking: cross-encoder reranker, MMR, reciprocal rank fusion, or domain-specific reranking.
7. Context assembly: citation metadata, deduplication, context packing, token-budget handling, source priority.
8. Generation: answer prompt, groundedness constraints, fallback behavior when retrieval confidence is low.
9. Evaluation: separate retrieval evaluation from final answer evaluation. Retrieval quality should be evaluated with recall@k, MRR, nDCG, coverage, and latency. Final answer quality should be evaluated with groundedness, faithfulness, citation correctness, usefulness, and human review hooks. Explain that a bad final answer may come from weak retrieval, noisy context assembly, or generation errors.
10. Failure modes: stale documents, bad chunks, low recall, noisy top-k, missing metadata filters, duplicated context, hallucination from weak context, and permission leakage.

Useful implementation references by name: FAISS, Chroma, Pinecone, Weaviate, pgvector, BM25, SentenceTransformers, OpenAI embeddings, LangChain retrievers, LlamaIndex, cross-encoder reranking.
Do not only say "accuracy vs speed" or "data quality". Provide specific retrieval design choices, but keep ungrounded numbers and library choices framed as options, tunable ranges, or tradeoffs.
For RAG questions, explicitly distinguish the retrieval module from the downstream generation module so the answer does not imply that poor final output always means poor retrieval.
""".strip()

    return ""


def _normalize_role(role: Optional[str]) -> str:
    if not role:
        return "general"

    role = str(role).strip().lower().replace(" ", "_").replace("-", "_")

    role_aliases = {
        "project_manager": "product_manager",
        "pm": "pm",
        "product": "product_manager",
        "product_manager": "product_manager",
        "software_engineer": "engineer",
        "developer": "engineer",
        "engineering": "engineer",
        "engineer": "engineer",
        "business": "business",
        "business_owner": "business_owner",
        "executive": "business_owner",
        "founder": "business_owner",
        "general": "general",
    }
    return role_aliases.get(role, role)


def _safe_get_query_type(query_understanding: Dict[str, Any]) -> str:
    return str(query_understanding.get("query_type", "concept_explanation"))


def _extract_structured_profile(retrieved_background_package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not retrieved_background_package:
        return {}

    # The current codebase may use either `structured_profile` or direct profile-like keys.
    if isinstance(retrieved_background_package.get("structured_profile"), dict):
        return retrieved_background_package["structured_profile"]

    return {
        key: value
        for key, value in retrieved_background_package.items()
        if key
        in {
            "role_lens",
            "role",
            "technical_depth",
            "technical_level",
            "jargon_tolerance",
            "preferred_explanation_style",
            "goal",
            "current_projects",
            "strength_areas",
            "weak_areas",
        }
    }


def _extract_background_chunks(retrieved_background_package: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not retrieved_background_package:
        return []

    chunks = retrieved_background_package.get("retrieved_background_chunks")
    if isinstance(chunks, list):
        return chunks

    chunks = retrieved_background_package.get("chunks")
    if isinstance(chunks, list):
        return chunks

    return []


def _apply_profile_preferences(plan: Dict[str, Any], structured_profile: Dict[str, Any]) -> Dict[str, Any]:
    preferences = structured_profile.get("preferred_explanation_style", []) or []
    if isinstance(preferences, str):
        preferences = [preferences]
    preferences = {str(p).lower() for p in preferences}

    if "step_by_step" in preferences or "step-by-step" in preferences:
        plan["structure_style"] = "step_by_step"

    if "analogy_driven" in preferences or "analogy" in preferences:
        plan["use_analogy"] = True

    if "concise" in preferences:
        plan["detail_level"] = "low_to_medium"
        if plan.get("target_audience") == "business_owner":
            plan["final_format"] = "concise executive summary with decision takeaway"
        elif plan.get("target_audience") == "product_manager":
            plan["final_format"] = "concise product summary with workflow and PM takeaway"
        else:
            plan["final_format"] = "concise structured answer"

    if "technical" in preferences:
        plan["detail_level"] = "high"
        plan["jargon_policy"] = "allow_technical_terms_with_context"
        plan["structure_style"] = "implementation_layered"
        plan["emphasis"] = list(dict.fromkeys(
            plan.get("emphasis", [])
            + ["architecture", "interfaces", "implementation details", "tradeoffs"]
        ))

    if "high_level" in preferences:
        plan["detail_level"] = "medium"
        if plan.get("target_audience") != "engineer":
            if plan.get("jargon_policy") in {"allow_technical_terms", "allow_technical_terms_with_context"}:
                plan["jargon_policy"] = "define_if_used"

    jargon_tolerance = str(structured_profile.get("jargon_tolerance", "")).lower()
    if jargon_tolerance == "low":
        plan["jargon_policy"] = "minimize_and_define_if_used"
    elif jargon_tolerance == "high":
        plan["jargon_policy"] = "allow_technical_terms_with_context"

    technical_depth = str(
        structured_profile.get("technical_depth")
        or structured_profile.get("technical_level")
        or ""
    ).lower()
    if technical_depth in {"low", "beginner"}:
        plan["detail_level"] = "low_to_medium"
        plan["jargon_policy"] = "minimize_and_define_if_used"
        plan["use_analogy"] = True
    elif technical_depth in {"high", "advanced"}:
        plan["detail_level"] = "high"
        plan["jargon_policy"] = "allow_technical_terms_with_context"
        plan["structure_style"] = "implementation_layered"
        plan["emphasis"] = list(dict.fromkeys(
            plan.get("emphasis", [])
            + ["architecture", "module interaction", "interfaces", "implementation details"]
        ))

    return plan


def _apply_background_chunk_signals(plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    chunk_text = "\n".join(str(chunk.get("text", "")) for chunk in chunks).lower()

    if any(term in chunk_text for term in ["not comfortable", "confused", "struggle", "weak"]):
        plan["jargon_policy"] = "minimize_and_define_if_used"
        plan["use_analogy"] = True
        if "known knowledge boundary" not in plan["emphasis"]:
            plan["emphasis"].append("explain around the user's known knowledge boundary")

    if any(term in chunk_text for term in ["business", "stakeholder", "management", "client"]):
        plan["include_business_impact"] = True
        if "stakeholder relevance" not in plan["emphasis"]:
            plan["emphasis"].append("stakeholder relevance")

    if any(term in chunk_text for term in ["product manager", "pm", "product", "roadmap", "requirement", "requirements", "user story", "stakeholder"]):
        if plan.get("target_audience") == "product_manager":
            plan["structure_style"] = "product_workflow_layered"
            plan["include_business_impact"] = True
            plan["emphasis"] = list(dict.fromkeys(
                plan.get("emphasis", [])
                + [
                    "user problem",
                    "workflow",
                    "requirements",
                    "dependencies",
                    "risks",
                    "success metrics",
                    "PM takeaway",
                ]
            ))

    if any(term in chunk_text for term in ["business owner", "executive", "leadership", "roi", "cost", "revenue", "operation", "customer impact"]):
        if plan.get("target_audience") == "business_owner":
            plan["structure_style"] = "executive_decision_summary"
            plan["include_business_impact"] = True
            plan["jargon_policy"] = "avoid_jargon"
            plan["emphasis"] = list(dict.fromkeys(
                plan.get("emphasis", [])
                + [
                    "business value",
                    "customer or operational impact",
                    "cost and risk implications",
                    "decision points",
                    "next step",
                ]
            ))

    if any(term in chunk_text for term in ["step-by-step", "step by step", "analogy"]):
        plan["structure_style"] = "step_by_step"
        plan["use_analogy"] = True

    if any(term in chunk_text for term in ["engineer", "developer", "backend", "api", "architecture", "implementation", "code"]):
        plan["detail_level"] = "high"
        plan["structure_style"] = "implementation_layered"
        plan["jargon_policy"] = "allow_technical_terms_with_context"
        plan["use_analogy"] = False
        plan["emphasis"] = list(dict.fromkeys(
            plan.get("emphasis", [])
            + [
                "architecture",
                "module responsibilities",
                "interfaces",
                "data flow",
                "implementation steps",
                "edge cases",
            ]
        ))

    return plan


def build_expression_plan(
    query_understanding: Dict[str, Any],
    retrieved_background_package: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    base_explanation: Optional[str] = None,
    use_llm_refinement: bool = False,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    """Build a structured expression plan.

    Args:
        query_understanding: Output from the query orchestrator.
        retrieved_background_package: Output from background retrieval.
        role: Optional UI-selected role. If provided, it overrides inferred role.
        base_explanation: Optional neutral answer used only for LLM refinement.
        use_llm_refinement: Whether to ask an LLM to refine the rule-based plan.
        model: LLM model name used for optional refinement.

    Returns:
        ExpressionPlan as a dictionary.
    """

    structured_profile = _extract_structured_profile(retrieved_background_package)
    background_chunks = _extract_background_chunks(retrieved_background_package)

    selected_role = role or structured_profile.get("role_lens") or structured_profile.get("role")
    normalized_role = _normalize_role(selected_role)

    role_defaults = ROLE_DEFAULTS.get(normalized_role, ROLE_DEFAULTS["general"]).copy()
    query_type = _safe_get_query_type(query_understanding)

    plan = ExpressionPlan(**role_defaults).to_dict()
    plan["explanation_goal"] = QUERY_TO_GOAL.get(query_type, "general_understanding")

    query_adjustment = QUERY_ADJUSTMENTS.get(query_type, {})
    if query_adjustment:
        for key, value in query_adjustment.items():
            if key == "emphasis":
                merged = list(dict.fromkeys(plan.get("emphasis", []) + value))
                plan["emphasis"] = merged
            else:
                plan[key] = value

    plan = _apply_profile_preferences(plan, structured_profile)
    plan = _apply_background_chunk_signals(plan, background_chunks)

    if use_llm_refinement:
        return refine_expression_plan_with_llm(
            initial_plan=plan,
            query_understanding=query_understanding,
            structured_profile=structured_profile,
            retrieved_background_chunks=background_chunks,
            base_explanation=base_explanation,
            model=model,
        )

    return plan


def refine_expression_plan_with_llm(
    initial_plan: Dict[str, Any],
    query_understanding: Dict[str, Any],
    structured_profile: Dict[str, Any],
    retrieved_background_chunks: List[Dict[str, Any]],
    base_explanation: Optional[str] = None,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    """Optionally refine the rule-based expression plan with an LLM."""

    if OpenAI is None:
        return initial_plan

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return initial_plan

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an expression planner for a personalized AI agent.

Your task is NOT to answer the user's question.
Your task is to refine the expression plan so that the final answer is best suited to the user's role, background, and communication preference.

Query understanding:
{json.dumps(query_understanding, indent=2)}

Structured profile:
{json.dumps(structured_profile, indent=2)}

Retrieved background chunks:
{json.dumps(retrieved_background_chunks, indent=2)}

Base explanation:
{base_explanation or "Not provided"}

Initial expression plan:
{json.dumps(initial_plan, indent=2)}

Return JSON only. Keep exactly these keys:
- target_audience
- explanation_goal
- detail_level
- tone
- structure_style
- use_analogy
- jargon_policy
- emphasis
- de_emphasis
- include_business_impact
- include_example
- example_style
- final_format
- grounding_policy
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
           
        )
        content = response.choices[0].message.content or ""
        refined = json.loads(content)
        return {**initial_plan, **refined}
    except Exception:
        return initial_plan


def rewrite_with_expression_plan(
    base_explanation: str,
    expression_plan: Dict[str, Any],
    query_understanding: Optional[Dict[str, Any]] = None,
    retrieved_background_package: Optional[Dict[str, Any]] = None,
    model: str = "gpt-5.5",
) -> str:
    """Rewrite a neutral/base explanation according to an ExpressionPlan.

    If the OpenAI client or API key is unavailable, this function returns a simple
    deterministic fallback so the app can still run locally.
    """

    if not base_explanation:
        return "I need a base explanation before I can rewrite it."

    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return _fallback_rewrite(base_explanation, expression_plan)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    role_output_contract = _get_role_output_contract(
        expression_plan,
        query_understanding=query_understanding,
    )
    topic_specific_engineer_guidance = _get_topic_specific_engineer_guidance(query_understanding)

    prompt = f"""
You are the expression layer of a personalized AI agent.

Rewrite the base explanation for the target user using the provided ExpressionPlan.
Keep the answer faithful to the base explanation and do not invent unsupported facts.

Query understanding:
{json.dumps(query_understanding or {}, indent=2)}

Retrieved background package:
{json.dumps(retrieved_background_package or {}, indent=2)}

ExpressionPlan:
{json.dumps(expression_plan, indent=2)}

Audience-specific output contract:
{role_output_contract}

Topic-specific implementation guidance:
{topic_specific_engineer_guidance or "No additional topic-specific guidance."}

Base explanation:
{base_explanation}

Rewrite requirements:
1. Follow the target audience, tone, detail level, and structure style.
2. Emphasize the items in `emphasis`.
3. De-emphasize or avoid the items in `de_emphasis`.
4. Follow the jargon policy.
5. Use analogy only if `use_analogy` is true.
6. Include an example only if `include_example` is true, and keep the example short and aligned with the target audience.
7. If business impact is requested, explain why it matters in practical terms, but do not invent unsupported ROI, cost, timeline, or customer-impact claims.
8. Follow the Audience-specific output contract. For project document answers, treat the contract as artifact-aware guidance and choose the section headings that best fit the project instead of forcing every listed section.
9. If topic-specific implementation guidance is provided, incorporate it into the answer when relevant.
10. For engineer-facing answers, avoid generic implementation notes. Prefer concrete methods, design alternatives, tradeoffs, evaluation metrics, and failure modes.
11. Follow the grounding_policy in the ExpressionPlan. Do not invent exact model dimensions, chunk sizes, index types, costs, latency numbers, library choices, ROI, timelines, or customer-impact claims unless explicitly supported by the base explanation, retrieved background, or config. If uncertain, phrase them as options, tunable ranges, or tradeoffs.
12. Do not append a second answer for another audience. The final output must target only one audience.
13. For non-project answers, use the exact section names required by the Audience-specific output contract. For project document answers, use clear artifact-appropriate section names from the contract or closely related names that fit the evidence.
13a. If this is a project document answer, explain the actual project details instead of forcing a generic system-design template. Do not use sections like Components, Interfaces, Retrieval / data-control flow, or Useful implementation references unless the project is actually about software architecture, RAG, or retrieval systems.
13b. For engineer-facing project document answers, preserve depth from the base explanation. If the base explanation contains multiple workflows, metrics, model roles, failure modes, or next steps, do not compress them into a short generic summary.
13c. For engineer-facing project document answers, explicitly separate architecture/workflow, data/features, model roles, evaluation metrics, technical risks, and next steps when the evidence supports those distinctions.
14. Return only the final personalized explanation.
""".strip()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You rewrite explanations clearly and faithfully."},
                {"role": "user", "content": prompt},
            ],
            
        )
        return response.choices[0].message.content or _fallback_rewrite(base_explanation, expression_plan)
    except Exception:
        return _fallback_rewrite(base_explanation, expression_plan)


def generate_personalized_explanation(
    base_explanation: str,
    query_understanding: Dict[str, Any],
    retrieved_background_package: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    use_llm_plan_refinement: bool = False,
    model: str = "gpt-5.5",
) -> Dict[str, Any]:
    """Build an expression plan and rewrite the base explanation.

    Returns a dictionary that can be displayed in Streamlit debug mode:
    - base_explanation
    - expression_plan
    - final_explanation
    """

    expression_plan = build_expression_plan(
        query_understanding=query_understanding,
        retrieved_background_package=retrieved_background_package,
        role=role,
        base_explanation=base_explanation,
        use_llm_refinement=use_llm_plan_refinement,
        model=model,
    )

    final_explanation = rewrite_with_expression_plan(
        base_explanation=base_explanation,
        expression_plan=expression_plan,
        query_understanding=query_understanding,
        retrieved_background_package=retrieved_background_package,
        model=model,
    )

    quality_report = evaluate_expression_quality(
        final_explanation=final_explanation,
        expression_plan=expression_plan,
        query_understanding=query_understanding,
    )

    return {
        "base_explanation": base_explanation,
        "query_understanding": query_understanding,
        "retrieved_background_package": retrieved_background_package or {},
        "expression_plan": expression_plan,
        "final_explanation": final_explanation,
        "quality_report": quality_report,
    }


def _fallback_rewrite(base_explanation: str, expression_plan: Dict[str, Any]) -> str:
    """Small deterministic fallback used when the LLM is unavailable."""

    audience = expression_plan.get("target_audience", "general")
    jargon_policy = expression_plan.get("jargon_policy", "define_if_used")
    emphasis = expression_plan.get("emphasis", [])
    body = base_explanation.strip()

    if audience == "engineer":
        sections = [
            "Project overview",
            body,
            "\nSystem / workflow architecture",
            "Reconstruct the pipeline from inputs to outputs when supported by the retrieved evidence.",
            "\nData sources and target",
            "Describe data sources, important inputs, features, and target variables only when supported by the retrieved evidence.",
            "\nModel / module roles",
            "Separate different models, modules, or layers by responsibility instead of collapsing them into one selected model.",
            "\nEvaluation results and metrics",
            "Include actual metrics and interpret whether they imply strong or weak performance when available.",
            "\nTechnical risks and limitations",
            "Discuss validation gaps, noisy signals, data leakage risks, underperformance, missing data, or production-readiness issues when supported by evidence.",
            "\nEngineering takeaway / next steps",
            "End with concrete reproduction, validation, monitoring, debugging, data enrichment, model tuning, or pipeline-improvement steps.",
        ]
        return "\n".join(sections)

    if audience == "product_manager":
        sections = [
            "1. Project overview",
            body,
            "\n2. User problem / product goal",
            "Translate the project into the user need, decision workflow, or analytical goal it supports.",
            "\n3. Workflow",
            "Describe the end-to-end flow from data to model/results in simple steps.",
            "\n4. Key result or selected model",
            "Explain the main finding, output, or selected model in product-friendly language.",
            "\n5. Risks / open questions",
            "Explain practical risks, unclear requirements, validation gaps, user-expectation risks, or operational concerns.",
            "\n6. PM takeaway and next step",
            "Explain whether this is an MVP, prototype, or analysis and what should happen next.",
        ]
        return "\n".join(sections)

    if audience == "business_owner":
        sections = [
            "1. Business takeaway",
            body,
            "\n2. What the project does",
            "Explain the project in plain language.",
            "\n3. Value / expected outcome",
            "Explain what business, operational, analytical, or decision value the project could create if supported by the evidence.",
            "\n4. Key result",
            "Summarize the most important result, selected model, or finding without unnecessary technical detail.",
            "\n5. Risks and limitations",
            "Explain what could make the result unreliable, hard to operationalize, or risky to over-trust.",
            "\n6. Recommended next step",
            "End with a practical leadership or stakeholder next step.",
        ]
        return "\n".join(sections)

    sections = [
        "1. Short definition",
        body,
        "\n2. How it works",
        "The system builds an expression plan from the user role, query type, and background context, then rewrites the base explanation.",
        "\n3. Why it matters",
        "This makes personalization explicit, inspectable, and easier to debug.",
    ]

    if emphasis:
        sections.append("\nKey focus: " + ", ".join(emphasis[:4]) + ".")

    if jargon_policy in {"avoid_jargon", "minimize_and_define_if_used"}:
        sections.append("\nTechnical terms should be minimized or defined when they appear.")

    return "\n".join(sections)