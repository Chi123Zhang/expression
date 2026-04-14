import os
import json
from typing import Dict, List
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _parse_json_safely(text: str) -> Dict:
    if not text:
        raise ValueError("Empty response from model.")

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


def _is_potentially_ambiguous_query(raw_query: str) -> Dict:
    q = raw_query.strip().lower().replace("?", "")

    ambiguous_terms = {
        "rag": {
            "topic": "RAG",
            "question": (
                'Do you mean "Retrieval-Augmented Generation" in AI, '
                'or "Red-Amber-Green" in project/status reporting?'
            )
        },
        "orchestrator": {
            "topic": "orchestrator",
            "question": (
                "Do you mean an orchestrator in AI/agent systems, "
                "or a project-specific orchestrator in your uploaded documents?"
            )
        },
        "api gateway": {
            "topic": "API gateway",
            "question": (
                "Do you want a general software architecture explanation of API Gateway, "
                "or are you referring to a specific project component?"
            )
        },
    }

    for term, meta in ambiguous_terms.items():
        if q == term or q == f"what is {term}":
            return {
                "is_ambiguous": True,
                "topic": meta["topic"],
                "question": meta["question"]
            }

    return {
        "is_ambiguous": False,
        "topic": None,
        "question": None
    }


def _default_background_chunk_types(query_type: str) -> List[str]:
    if query_type == "concept_explanation":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference"
        ]
    if query_type == "comparison_question":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference",
            "role_identity"
        ]
    if query_type in {"project_explanation", "document_based_question"}:
        return [
            "current_project",
            "role_identity",
            "expression_preference"
        ]
    if query_type == "workflow_explanation":
        return [
            "knowledge_boundary",
            "expression_preference",
            "role_identity",
            "current_project"
        ]
    return [
        "role_identity",
        "technical_exposure",
        "expression_preference"
    ]


def _smart_chunk_selection(query_type: str, topic: str, intent: str, domain: str) -> List[str]:
    """
    Smarter chunk selection for Person 2.
    This is the main fix for problem 1.
    """
    qt = (query_type or "").lower()
    tp = (topic or "").lower()
    it = (intent or "").lower()
    dm = (domain or "").lower()

    # Project / uploaded-doc style questions
    if qt in {"project_explanation", "document_based_question"}:
        return [
            "current_project",
            "role_identity",
            "expression_preference"
        ]

    # Workflow / architecture style questions
    if qt == "workflow_explanation":
        return [
            "knowledge_boundary",
            "expression_preference",
            "technical_exposure",
            "role_identity"
        ]

    # Comparison questions
    if qt == "comparison_question":
        return [
            "technical_exposure",
            "knowledge_boundary",
            "expression_preference",
            "role_identity"
        ]

    # Concept explanations
    if qt == "concept_explanation":
        # Orchestrator / architecture-like concept
        if any(x in tp for x in ["orchestrator", "system", "architecture", "agent"]) or \
           any(x in it for x in ["role", "function", "workflow"]) or \
           "artificial intelligence" in dm or "ai" in dm:
            return [
                "knowledge_boundary",
                "expression_preference",
                "technical_exposure"
            ]

        # Definitions like RAG / vector DB / API gateway
        if any(x in tp for x in ["rag", "retrieval", "vector", "database", "api", "gateway"]):
            return [
                "technical_exposure",
                "expression_preference",
                "knowledge_boundary"
            ]

        # Generic concept fallback
        return [
            "technical_exposure",
            "expression_preference",
            "domain_context"
        ]

    return _default_background_chunk_types(qt)


def understand_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    Person 2 - Query Understanding
    Produces a Query Understanding Object.
    """

    raw_query = raw_query.strip()

    # -----------------------------
    # Rule-based ambiguity first pass
    # -----------------------------
    ambiguity = _is_potentially_ambiguous_query(raw_query)
    if ambiguity["is_ambiguous"]:
        return {
            "query_id": "q_auto",
            "user_id": user_id,
            "raw_query": raw_query,
            "query_type": "clarification_needed",
            "topic": ambiguity["topic"],
            "subtopics": [],
            "intent": "resolve_ambiguity",
            "domain": "",
            "requires_background_retrieval": False,
            "requires_project_context": False,
            "requires_external_knowledge": False,
            "needs_clarification": True,
            "clarification_reason": "The query term is ambiguous and has multiple plausible meanings.",
            "suggested_clarification_question": ambiguity["question"],
            "recommended_background_chunk_types": [],
            "recommended_next_step": "clarification",
        }

    client = _get_openai_client()

    prompt = f"""
You are the query understanding module of a personalized explanation agent.

Your job is to classify the user's query and decide what type of context is needed.

Return ONLY valid JSON with exactly these keys:

- query_id
- user_id
- raw_query
- query_type
- topic
- subtopics
- intent
- domain
- requires_background_retrieval
- requires_project_context
- requires_external_knowledge
- needs_clarification
- clarification_reason
- suggested_clarification_question
- recommended_background_chunk_types
- recommended_next_step

Allowed values:
- query_type must be one of:
  ["concept_explanation", "project_explanation", "comparison_question", "workflow_explanation", "document_based_question", "clarification_needed"]

- recommended_background_chunk_types must be chosen from:
  ["role_identity", "domain_context", "technical_exposure", "knowledge_boundary", "expression_preference", "current_project"]

- recommended_next_step must be one of:
  ["clarification", "retrieve_background_then_explain", "retrieve_background_and_project_then_explain", "external_knowledge_then_explain"]

Important routing rules:
1. If the user asks a general concept question like:
   - What is retrieval-augmented generation?
   - What is a vector database?
   - Explain API gateway
   - Explain what an orchestrator does in an AI agent system
   then this is usually:
   - query_type = "concept_explanation"
   - requires_external_knowledge = true
   - requires_project_context = false

2. Only set requires_project_context = true when the question clearly depends on uploaded project documents, such as:
   - Explain this project
   - Explain this architecture in the uploaded note
   - What is the study design in the uploaded document?
   - Summarize the uploaded note
   - What does this project document say about X?

3. If the question is vague and depends on missing context, set:
   - needs_clarification = true

4. Background retrieval is usually useful for personalization.

Current context:
- user_id = "{user_id}"
- has_uploaded_project_doc = {str(has_uploaded_project_doc)}
- raw_query = "{raw_query}"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    result = _parse_json_safely(content)

    valid_query_types = {
        "concept_explanation",
        "project_explanation",
        "comparison_question",
        "workflow_explanation",
        "document_based_question",
        "clarification_needed",
    }

    valid_next_steps = {
        "clarification",
        "retrieve_background_then_explain",
        "retrieve_background_and_project_then_explain",
        "external_knowledge_then_explain",
    }

    valid_chunk_types = {
        "role_identity",
        "domain_context",
        "technical_exposure",
        "knowledge_boundary",
        "expression_preference",
        "current_project",
    }

    if result.get("query_type") not in valid_query_types:
        result["query_type"] = "concept_explanation"

    if not isinstance(result.get("subtopics"), list):
        result["subtopics"] = []

    for key in [
        "requires_background_retrieval",
        "requires_project_context",
        "requires_external_knowledge",
        "needs_clarification",
    ]:
        if not isinstance(result.get(key), bool):
            result[key] = False

    # -----------------------------
    # SMART FIX FOR PROBLEM 1
    # Override chunk selection with smarter logic
    # -----------------------------
    smart_chunks = _smart_chunk_selection(
        query_type=result.get("query_type", ""),
        topic=result.get("topic", ""),
        intent=result.get("intent", ""),
        domain=result.get("domain", "")
    )

    result["recommended_background_chunk_types"] = [
        x for x in smart_chunks if x in valid_chunk_types
    ]

    if not result["recommended_background_chunk_types"]:
        result["recommended_background_chunk_types"] = _default_background_chunk_types(
            result.get("query_type", "concept_explanation")
        )

    if result.get("recommended_next_step") not in valid_next_steps:
        if result.get("needs_clarification", False):
            result["recommended_next_step"] = "clarification"
        elif result.get("requires_external_knowledge", False):
            result["recommended_next_step"] = "external_knowledge_then_explain"
        elif result.get("requires_project_context", False):
            result["recommended_next_step"] = "retrieve_background_and_project_then_explain"
        else:
            result["recommended_next_step"] = "retrieve_background_then_explain"

    result["user_id"] = user_id
    result["raw_query"] = raw_query

    if not result.get("query_id"):
        result["query_id"] = "q_auto"

    return result


def route_query(query_understanding_object: Dict) -> Dict:
    """
    Person 2 - Routing decision
    """

    if query_understanding_object.get("needs_clarification", False):
        return {
            "route": "clarification",
            "message": query_understanding_object.get(
                "suggested_clarification_question",
                "Could you clarify what kind of explanation you want?"
            )
        }

    if query_understanding_object.get("requires_external_knowledge", False):
        return {
            "route": "external_knowledge_then_expression",
            "background_request": {
                "user_id": query_understanding_object["user_id"],
                "query": query_understanding_object["raw_query"],
                "recommended_background_chunk_types": query_understanding_object.get(
                    "recommended_background_chunk_types", []
                )
            }
        }

    if query_understanding_object.get("requires_project_context", False):
        return {
            "route": "background_and_project_then_expression",
            "background_request": {
                "user_id": query_understanding_object["user_id"],
                "query": query_understanding_object["raw_query"],
                "recommended_background_chunk_types": query_understanding_object.get(
                    "recommended_background_chunk_types", []
                )
            }
        }

    return {
        "route": "background_retrieval_then_expression",
        "background_request": {
            "user_id": query_understanding_object["user_id"],
            "query": query_understanding_object["raw_query"],
            "recommended_background_chunk_types": query_understanding_object.get(
                "recommended_background_chunk_types", []
            )
        }
    }


def process_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    One-step wrapper:
    returns both query understanding and routing decision.
    """
    q_obj = understand_query(
        user_id=user_id,
        raw_query=raw_query,
        has_uploaded_project_doc=has_uploaded_project_doc
    )

    routing = route_query(q_obj)

    return {
        "query_understanding_object": q_obj,
        "routing_decision": routing
    }
