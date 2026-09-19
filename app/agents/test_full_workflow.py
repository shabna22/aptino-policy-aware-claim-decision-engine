import json
import os

from app.agents.state import ClaimState
from app.agents.case_analysis_agent import case_analysis_agent
from app.agents.policy_evidence_agent import policy_evidence_agent
from app.agents.coverage_agent import coverage_exclusion_agent
from app.agents.decision_agent import decision_agent
from app.agents.validation_agent import validation_agent


INPUT_FILE = "candidate_data/public_test_cases.json"
OUTPUT_FILE = "public_evaluation_results.json"


# Load public test cases
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    cases = json.load(file)


# Load existing results if the file already exists.
# This allows the evaluation to resume if it stops partway through.
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)
else:
    results = []


# Keep only completed case IDs for resume functionality
completed_case_ids = {
    result["case_id"]
    for result in results
}


# Process each claim exactly once
for claim in cases:

    case_id = claim["case_id"]

    # Skip cases that have already been completed
    if case_id in completed_case_ids:
        print(f"Skipping {case_id} - already completed.")
        continue

    print("\n" + "=" * 70)
    print(f"Running {case_id}")
    print("=" * 70)

    try:

        # Create a fresh state for the current claim
        state: ClaimState = {
            "case_id": case_id,
            "claim": claim,
            "trace": []
        }

        # Run the multi-agent workflow
        state = case_analysis_agent(state)
        state = policy_evidence_agent(state)
        state = coverage_exclusion_agent(state)
        state = decision_agent(state)
        state = validation_agent(state)

        # Store the final result
        result = {
            "case_id": state["case_id"],
            "decision": state["decision"],
            "validation": state["validation"],
            "trace": state["trace"]
        }

        results.append(result)

        # Mark this case as completed
        completed_case_ids.add(case_id)

        # Save immediately after every successful case
        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(results, file, indent=2)

        print("\nRESULT:")
        print(
            f"Decision: "
            f"{state['decision'].get('decision')}"
        )

        print(
            f"Confidence: "
            f"{state['decision'].get('confidence')}"
        )

        print(
            f"Validation: "
            f"{state['validation'].get('validation_status')}"
        )

        print(f"\nSaved {case_id} successfully.")

    except Exception as error:

        print("\n" + "!" * 70)
        print(f"ERROR while processing {case_id}")
        print("!" * 70)

        print(type(error).__name__)
        print(str(error))

        print(
            "\nAlready completed cases have been saved."
        )

        print(
            f"Results file: {OUTPUT_FILE}"
        )

        break


# Final evaluation summary
print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

for result in results:

    print(
        f"{result['case_id']}: "
        f"{result['decision'].get('decision')} | "
        f"Validation: "
        f"{result['validation'].get('validation_status')}"
    )


# Count unique completed cases rather than raw result entries
unique_completed_cases = {
    result["case_id"]
    for result in results
}

print(
    f"\nCompleted cases: "
    f"{len(unique_completed_cases)}/{len(cases)}"
)

print(
    f"Complete results saved to: {OUTPUT_FILE}"
)