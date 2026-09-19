import json

from app.agents.state import ClaimState
from app.agents.case_analysis_agent import case_analysis_agent
from app.agents.policy_evidence_agent import policy_evidence_agent


with open(
    "candidate_data/public_test_cases.json",
    "r",
    encoding="utf-8"
) as file:
    cases = json.load(file)


# Use the first public case
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


print("\nPOLICY EVIDENCE AGENT")
print("=" * 60)

for evidence in state["retrieved_evidence"]:

    print(f"\nChunk ID: {evidence['chunk_id']}")
    print(f"Page: {evidence['page']}")
    print(f"Section: {evidence['section']}")
    print(
        f"Reranker Score: "
        f"{evidence['reranker_score']:.4f}"
    )
    print(
        f"Text:\n"
        f"{evidence['text'][:500]}"
    )
    print("-" * 60)


print("\nTRACE")
print(json.dumps(
    state["trace"],
    indent=2
))