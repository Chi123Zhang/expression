import os
import json
from typing import Dict, Optional
from openai import OpenAI


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def understand_query(
    user_id: str,
    raw_query: str,
    has_uploaded_project_doc: bool = False
) -> Dict:
    """
    Person 2 - Step 1~7:
    Build the Query Understanding Object.
    """

    client = _get_openai_client()

    prompt = f"""
You are the query understanding module of a personalized explanation agent.

Analyze the user query and return ONLY valid JSON with exactly these keys:

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

Rules:
- query_type must be one of:
  ["concept_explanation", "project_explanation", "comparison_question", "workflow_explanation", "document_based_question", "clarification_needed"]
- subtopics must be a list of strings
- requires_background_retrieval / requires_project_context / requires_external_knowledge / needs_clarification must be booleans
- recommended_background_chunk_types must be chosen from:
  ["role_identity", "domain_context", "technical_exposure", "knowledge_boundary", "expression_preference", "current_project"]
- recommended_next_step must be one of:
  ["clarification", "retrieve_background_then_explain", "retrieve_background_and_project_then_explain", "external_knowledge_then_explain"]

Context:
- user_id = "{user_id}"
- has_uploaded_project_doc = {str(has_uploaded_project_doc)}
- raw_query = "{raw_query}"

Interpretation guideline:
- If the query is about a concept like "What is RAG?" or "What is an orchestrator?", it is likely concept_explanation.
- If the query refers to "this project", "this architecture", or uploaded notes/docs, it may require project context.
- If the query is too vague (e.g. "Explain this") and no project doc exists, clarification is needed.
- In this system, user background retrieval is usually useful unless the query is extremely generic.
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
    result = json.loads(content)

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

    if not isinstance(result.get("recommended_background_chunk_types"), list):
        result["recommended_background_chunk_types"] = []

    result["recommended_background_chunk_types"] = [
        x for x in result["recommended_background_chunk_types"]
        if x in valid_chunk_types
    ]

    if result.get("recommended_next_step") not in valid_next_steps:
        result["recommended_next_step"] = "retrieve_background_then_explain"

    result["user_id"] = user_id
    result["raw_query"] = raw_query
    if not result.get("query_id"):
        result["query_id"] = "q_auto"

    return result


def route_query(query_understanding_object: Dict) -> Dict:
    """
    Person 2 - Step 8:
    Make routing decision from Query Understanding Object.
    """

    if query_understanding_object.get("needs_clarification", False):
        return {
            "route": "clarification",
            "message": query_understanding_object.get(
                "suggested_clarification_question",
                "Could you clarify what kind of explanation you want?"
            )
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
    One-step wrapper for Person 2.
    Returns both query understanding and routing decision.
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
