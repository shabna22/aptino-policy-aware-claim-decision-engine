import json
from pathlib import Path
from collections import Counter


ROOT = Path(__file__).resolve().parent.parent

PUBLIC_RESULTS = ROOT / "public_evaluation_results.json"
ADDITIONAL_RESULTS = ROOT / "additional_evaluation_results.json"


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(results):
    total = len(results)

    decisions = Counter()

    for result in results:
        decision = result.get("decision", "UNKNOWN")

        if isinstance(decision, dict):
           decision = decision.get("status") or decision.get("decision") or "UNKNOWN"

        decisions[str(decision)] += 1

    validation = Counter()

    for result in results:
        validation_data = result.get("validation", {})

    if isinstance(validation_data, dict):
        status = validation_data.get("status", "UNKNOWN")
    else:
        status = str(validation_data)

    validation[str(status)] += 1

    needs_review = decisions.get("NEEDS_REVIEW", 0)

    total_findings = 0
    cited_findings = 0

    for result in results:
        findings = result.get("key_findings", [])

        for finding in findings:
            total_findings += 1

            if finding.get("citations"):
                cited_findings += 1

    citation_coverage = (
        cited_findings / total_findings
        if total_findings
        else 0
    )

    validation_pass_rate = (
        validation.get("PASS", 0) / total
        if total
        else 0
    )

    return {
        "total_cases": total,
        "decision_distribution": dict(decisions),
        "needs_review_cases": needs_review,
        "validation_distribution": dict(validation),
        "validation_pass_rate": round(
            validation_pass_rate * 100,
            2
        ),
        "finding_citation_coverage": round(
            citation_coverage * 100,
            2
        ),
    }


def main():

    public_results = load_results(PUBLIC_RESULTS)
    additional_results = load_results(ADDITIONAL_RESULTS)

    all_results = public_results + additional_results

    public_metrics = evaluate(public_results)
    additional_metrics = evaluate(additional_results)
    overall_metrics = evaluate(all_results)

    output = {
        "public_evaluation": public_metrics,
        "additional_evaluation": additional_metrics,
        "overall": overall_metrics,
        "failure_cases": [
            {
                "case_id": "PUB-003",
                "observed_decision": "ADMISSIBLE",
                "expected_policy_outcome": "NOT_ADMISSIBLE",
                "root_cause": (
                    "The decision agent did not correctly apply "
                    "the 48-month continuous coverage requirement "
                    "for pre-existing diseases."
                ),
                "improvement": (
                    "Add a deterministic waiting-period guardrail "
                    "before final decision generation."
                ),
            },
            {
                "case_id": "PUB-012",
                "observed_decision": "NEEDS_REVIEW",
                "expected_policy_outcome": "NOT_ADMISSIBLE",
                "root_cause": (
                    "The decision agent abstained despite the "
                    "case explicitly identifying experimental "
                    "treatment."
                ),
                "improvement": (
                    "Add deterministic handling for explicit "
                    "experimental/unproven treatment exclusions."
                ),
            },
            {
                "case_id": "ADD-001",
                "observed_decision": "ADMISSIBLE",
                "expected_policy_outcome": "NOT_ADMISSIBLE",
                "root_cause": (
                    "The initial 30-day waiting period was not "
                    "correctly applied."
                ),
                "improvement": (
                    "Add a deterministic initial waiting-period "
                    "check before LLM decision generation."
                ),
            },
        ],
    }

    output_path = ROOT / "evaluation_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("POLICY-AWARE CLAIM ENGINE EVALUATION")
    print("=" * 60)

    print(f"\nPublic cases: {public_metrics['total_cases']}")
    print(
        f"Additional cases: "
        f"{additional_metrics['total_cases']}"
    )
    print(
        f"Total cases: "
        f"{overall_metrics['total_cases']}"
    )

    print("\nDecision distribution:")
    for decision, count in overall_metrics[
        "decision_distribution"
    ].items():
        print(f"  {decision}: {count}")

    print(
        f"\nNEEDS_REVIEW cases: "
        f"{overall_metrics['needs_review_cases']}"
    )

    print(
        f"Validation pass rate: "
        f"{overall_metrics['validation_pass_rate']}%"
    )

    print(
        f"Citation coverage: "
        f"{overall_metrics['finding_citation_coverage']}%"
    )

    print(
        f"\nResults written to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()