import json

from state import ClaimState
from case_analysis_agent import case_analysis_agent


with open(
    "candidate_data/public_test_cases.json",
    "r",
    encoding="utf-8"
) as file:
    cases = json.load(file)


# Test with the first public case
claim = cases[0]


state: ClaimState = {
    "case_id": claim["case_id"],
    "claim": claim,
    "trace": []
}


result = case_analysis_agent(state)


print("\nCASE ANALYSIS AGENT")
print("=" * 60)

print(json.dumps(
    result["case_analysis"],
    indent=2
))

print("\nTRACE")
print(json.dumps(
    result["trace"],
    indent=2
))