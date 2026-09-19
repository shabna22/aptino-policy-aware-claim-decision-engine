import json

from google import genai
from dotenv import load_dotenv

from app.agents.state import ClaimState


load_dotenv()

client = genai.Client()


def decision_agent(state: ClaimState) -> ClaimState:

    claim = state["claim"]
    case_analysis = state["case_analysis"]
    coverage_analysis = state["coverage_analysis"]
    evidence = state["retrieved_evidence"]

    evidence_summary = [
        {
            "chunk_id": item["chunk_id"],
            "page": item["page"],
            "section": item["section"],
            "text": item["text"],
        }
        for item in evidence
    ]

    prompt = f"""
You are the Decision Agent in a policy-aware
health-insurance claim decision system.

Use ONLY:
1. Claim facts
2. Case Analysis
3. Coverage & Exclusion Analysis
4. Retrieved policy evidence

Do not use outside insurance or medical knowledge.

IMPORTANT PROVENANCE RULES:

Every key finding must have a type:

CLAIM_FACT
- Information directly provided by the claim.

POLICY_RULE
- A rule or requirement stated in the policy.
- Must have at least one policy citation.

CALCULATION
- A calculation based on claim values and policy rules.
- Must cite the policy rule used.

INFERENCE
- A conclusion requiring interpretation.
- Must have supporting policy citations.

Do NOT treat a claim fact as requiring a policy citation.

If policy evidence is insufficient for a material policy
question, use NEEDS_REVIEW and reason:
INSUFFICIENT_EVIDENCE

Allowed decisions:

ADMISSIBLE
ADMISSIBLE_WITH_LIMITS
PARTIALLY_ADMISSIBLE
NOT_ADMISSIBLE
NEEDS_REVIEW

Return ONLY valid JSON.

Required structure:

{{
    "decision": "",
    "confidence": 0.0,
    "key_findings": [
        {{
            "statement": "",
            "type": "",
            "citations": []
        }}
    ],
    "applicable_limits": [],
    "missing_evidence": [],
    "citations": [],
    "reason": ""
}}

For citations use:

{{
    "source": "policy",
    "page": 0,
    "section": "",
    "chunk_id": ""
}}

CLAIM:
{json.dumps(claim, indent=2)}

CASE ANALYSIS:
{json.dumps(case_analysis, indent=2)}

COVERAGE AND EXCLUSION ANALYSIS:
{json.dumps(coverage_analysis, indent=2)}

RETRIEVED POLICY EVIDENCE:
{json.dumps(evidence_summary, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    decision = json.loads(response.text)

    state["decision"] = decision

    state.setdefault("trace", []).append(
        {
            "agent": "Decision Agent",
            "action": (
                "Generated a policy-grounded decision with "
                "claim/policy provenance."
            )
        }
    )

    return state