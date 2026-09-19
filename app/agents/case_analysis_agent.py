import json
from google import genai
from dotenv import load_dotenv

from app.agents.state import ClaimState


load_dotenv()

client = genai.Client()


def case_analysis_agent(state: ClaimState) -> ClaimState:
    """
    Analyze the claim and identify:
    - relevant facts
    - decision dimensions
    - missing evidence
    - investigation plan
    """

    claim = state["claim"]

    prompt = f"""
You are the Case Analysis Agent in a health-insurance
claim decision system.

Analyze ONLY the information provided in the claim below.

Do not use outside medical or insurance knowledge.

Identify:
1. Relevant facts
2. Decision dimensions that need to be checked
3. Missing evidence or unclear information
4. Investigation plan for the next agents

Return ONLY a JSON object.
Do not use Markdown.
Do not use ```json.
Do not add explanations outside the JSON.

Required structure:

{{
    "relevant_facts": [],
    "decision_dimensions": [],
    "missing_evidence": [],
    "investigation_plan": []
}}

CLAIM:
{json.dumps(claim, indent=2)}
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

    analysis = json.loads(response.text)

    state["case_analysis"] = analysis

    state.setdefault("trace", []).append(
        {
            "agent": "Case Analysis Agent",
            "action": (
                "Analyzed claim facts and identified "
                "decision dimensions and missing evidence."
            )
        }
    )

    return state