import json

from google import genai
from dotenv import load_dotenv

from app.agents.state import ClaimState


load_dotenv()

client = genai.Client()


def coverage_exclusion_agent(state: ClaimState) -> ClaimState:
    """
    Analyze coverage, waiting periods, exclusions,
    definitions, and applicable limits using only
    retrieved policy evidence.
    """

    claim = state["claim"]
    case_analysis = state["case_analysis"]
    evidence = state["retrieved_evidence"]

    evidence_text = []

    for item in evidence:
        evidence_text.append(
            f"""
CHUNK ID: {item['chunk_id']}
PAGE: {item['page']}
SECTION: {item['section']}
TEXT:
{item['text']}
"""
        )

    evidence_context = "\n".join(evidence_text)

    prompt = f"""
You are the Coverage and Exclusion Agent in a
health-insurance claim decision system.

Your job is to analyze the claim using ONLY the
policy evidence retrieved by the Policy Evidence Agent.

Do NOT use outside insurance or medical knowledge.

Determine:

1. Applicable coverage provisions
2. Applicable waiting periods
3. Applicable exclusions
4. Applicable definitions
5. Applicable limits or sub-limits
6. Whether the retrieved evidence is sufficient
7. Any additional evidence required

Do NOT make the final claim decision.

Every important policy statement must reference the
chunk_id and page from the supplied evidence.

Return ONLY valid JSON.

Required structure:

{{
    "applicable_coverage": [],
    "waiting_periods": [],
    "exclusions": [],
    "definitions": [],
    "limits": [],
    "evidence_sufficient": true,
    "additional_evidence_required": [],
    "policy_references": []
}}

CLAIM:
{json.dumps(claim, indent=2)}

CASE ANALYSIS:
{json.dumps(case_analysis, indent=2)}

RETRIEVED POLICY EVIDENCE:
{evidence_context}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    analysis = json.loads(response.text)

    state["coverage_analysis"] = analysis

    state.setdefault("trace", []).append(
        {
            "agent": "Coverage & Exclusion Agent",
            "action": (
                "Analyzed retrieved policy evidence for "
                "coverage, waiting periods, exclusions, "
                "definitions, and limits."
            )
        }
    )

    return state