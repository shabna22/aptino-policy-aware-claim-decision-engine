import json
import re

from google import genai
from dotenv import load_dotenv

from app.agents.state import ClaimState


load_dotenv()

client = genai.Client()


def calculate_expected_limit(statement, claim):
    """
    Deterministically verify percentage-based calculations
    using the Sum Insured from the claim.
    """

    sum_insured = claim.get("sum_insured")

    if sum_insured is None:
        return None

    try:
        sum_insured = float(sum_insured)
    except (TypeError, ValueError):
        return None

    # Find percentage such as 1.0%, 25%, 40%
    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        statement
    )

    if not percentage_match:
        return None

    percentage = float(
        percentage_match.group(1)
    )

    # Calculate the basic limit
    expected = sum_insured * percentage / 100

    # Check whether the calculation is per day
    days_match = re.search(
        r"per\s+day.*?for\s+(\d+)\s+days",
        statement,
        re.IGNORECASE
    )

    if days_match:
        days = int(days_match.group(1))
        expected = expected * days

    else:
        # Also support wording like:
        # "per day for 4 days"
        days_match = re.search(
            r"per\s+day.*?(\d+)\s+days",
            statement,
            re.IGNORECASE
        )

        if days_match:
            days = int(days_match.group(1))
            expected = expected * days

    # Find the amount stated after "equals"
    equals_match = re.search(
        r"equals\s+([\d,]+(?:\.\d+)?)",
        statement,
        re.IGNORECASE
    )

    if not equals_match:
        return None

    stated_amount = float(
        equals_match.group(1).replace(",", "")
    )

    return {
        "expected": expected,
        "stated": stated_amount,
        "correct": abs(expected - stated_amount) < 0.01
    }


def validation_agent(state: ClaimState) -> ClaimState:

    claim = state["claim"]
    decision = state["decision"]
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
You are the Validation Agent for a policy-aware
health-insurance claim system.

Validate the Decision Agent output.

There are THREE different things to validate:

1. CLAIM_FACT

Check whether the statement is actually present
in the supplied claim.

A CLAIM_FACT does NOT require a policy citation.

2. POLICY_RULE

Check whether the statement is supported by the
retrieved policy evidence.

Check whether its citation points to a relevant
policy chunk.

3. CALCULATION

Check whether the calculation is logically based
on a policy rule and claim values.

IMPORTANT:

Do NOT mark a CLAIM_FACT as unsupported merely because
it does not appear in the policy.

Do NOT require a policy citation for a CLAIM_FACT.

Do NOT independently invent policy rules.

For CALCULATION statements, the numerical arithmetic
will be checked separately by the application.

Therefore, focus your validation on whether the
calculation is based on an appropriate policy rule
and the supplied claim values.

Return FAIL only if a material decision statement is
unsupported or materially incorrect.

Return ONLY valid JSON.

Required structure:

{{
    "validation_status": "PASS",
    "supported_claims": [],
    "unsupported_claims": [],
    "citation_checks": [],
    "calculation_checks": [],
    "notes": ""
}}

CLAIM:
{json.dumps(claim, indent=2)}

DECISION:
{json.dumps(decision, indent=2)}

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

    try:
        validation = json.loads(response.text)
    except json.JSONDecodeError as error:
        print("\nValidation Agent returned invalid JSON:")
        print(response.text)
        raise ValueError(
             f"Validation Agent returned invalid JSON: {error}"
        )

    # ---------------------------------------------------------
    # Deterministic calculation validation
    # ---------------------------------------------------------

    calculation_results = []

    for finding in decision.get("key_findings", []):

        if finding.get("type") != "CALCULATION":
            continue

        statement = finding.get("statement", "")

        result = calculate_expected_limit(
            statement,
            claim
        )

        if result is None:
            # Let the LLM handle calculations that cannot
            # be parsed deterministically.
            continue

        calculation_results.append(
            {
                "statement": statement,
                "calculation_status": (
                    "PASS"
                    if result["correct"]
                    else "FAIL"
                ),
                "expected_value": result["expected"],
                "stated_value": result["stated"]
            }
        )

    # Add deterministic calculation checks
    validation["calculation_checks"] = calculation_results

    # ---------------------------------------------------------
    # Final validation status
    # ---------------------------------------------------------

    has_failed_calculation = any(
        item["calculation_status"] == "FAIL"
        for item in calculation_results
    )

    has_unsupported_claims = bool(
        validation.get("unsupported_claims")
    )

    has_failed_citation = any(
        item.get("citation_status") == "FAIL"
        for item in validation.get("citation_checks", [])
    )

    if (
        has_failed_calculation
        or has_unsupported_claims
        or has_failed_citation
    ):
        validation["validation_status"] = "FAIL"
    else:
        validation["validation_status"] = "PASS"

    # ---------------------------------------------------------
    # Trace
    # ---------------------------------------------------------

    state["validation"] = validation

    state.setdefault("trace", []).append(
        {
            "agent": "Validation Agent",
            "action": (
                "Validated claim facts, policy rules, "
                "citations, and calculations."
            ),
            "validation_status": validation.get(
                "validation_status"
            )
        }
    )

    return state