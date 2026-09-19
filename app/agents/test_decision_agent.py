import json

from app.agents.state import ClaimState
from app.agents.case_analysis_agent import case_analysis_agent
from app.agents.policy_evidence_agent import policy_evidence_agent
from app.agents.coverage_agent import coverage_exclusion_agent
from app.agents.decision_agent import decision_agent


with open(
    "candidate_data/public_test_cases.json",
    "r",
    encoding="utf-8"
) as file:
    cases = json.load(file)


claim = cases[0]


state: ClaimState = {
    "case_id": claim["case_id"],
    "claim": claim,
    "trace": []
}


# Agent 1
state = case_analysis_agent(state)

# Agent 2
state = policy_evidence_agent(state)

# Agent 3
state = coverage_exclusion_agent(state)

# Agent 4
state = decision_agent(state)


print("\nDECISION AGENT")
print("=" * 60)

print(
    json.dumps(
        state["decision"],
        indent=2
    )
)

print("\nTRACE")
print(
    json.dumps(
        state["trace"],
        indent=2
    )
)