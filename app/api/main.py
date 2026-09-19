from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict

from app.agents.state import ClaimState
from app.agents.case_analysis_agent import case_analysis_agent
from app.agents.policy_evidence_agent import policy_evidence_agent
from app.agents.coverage_agent import coverage_exclusion_agent
from app.agents.decision_agent import decision_agent
from app.agents.validation_agent import validation_agent


app = FastAPI(
    title="Policy-Aware Multi-Agent RAG Claim Decision Engine",
    version="1.0.0"
)


class AnalyzeRequest(BaseModel):
    claim: Dict[str, Any] = Field(
        ...,
        description="Synthetic health-insurance claim case"
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "policy-aware-claim-decision-engine"
    }


@app.post("/analyze")
def analyze_claim(request: AnalyzeRequest):

    claim = request.claim

    if not claim:
        raise HTTPException(
            status_code=400,
            detail="Claim data cannot be empty."
        )

    case_id = claim.get("case_id")

    if not case_id:
        raise HTTPException(
            status_code=400,
            detail="claim.case_id is required."
        )

    try:

        state: ClaimState = {
            "case_id": case_id,
            "claim": claim,
            "trace": []
        }

        # Agent 1: Case Analysis
        state = case_analysis_agent(state)

        # Agent 2: Policy Evidence Retrieval
        state = policy_evidence_agent(state)

        # Agent 3: Coverage and Exclusion Analysis
        state = coverage_exclusion_agent(state)

        # Agent 4: Decision
        state = decision_agent(state)

        # Agent 5: Validation
        state = validation_agent(state)

        decision = state.get("decision", {})
        validation = state.get("validation", {})

        return {
            "case_id": state["case_id"],
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "key_findings": decision.get("key_findings", []),
            "applicable_limits": decision.get(
                "applicable_limits", []
            ),
            "missing_evidence": decision.get(
                "missing_evidence", []
            ),
            "citations": decision.get(
                "citations", []
            ),
            "validation": validation,
            "trace": state.get("trace", [])
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Claim analysis failed: {str(error)}"
        )