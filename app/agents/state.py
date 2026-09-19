from typing import TypedDict, List, Dict, Any


class ClaimState(TypedDict, total=False):
    # Input claim
    case_id: str
    claim: Dict[str, Any]

    # Case Analysis Agent
    case_analysis: Dict[str, Any]

    # Policy Evidence Agent
    retrieved_evidence: List[Dict[str, Any]]

    # Coverage & Exclusion Agent
    coverage_analysis: Dict[str, Any]

    # Decision Agent
    decision: Dict[str, Any]

    # Validation Agent
    validation: Dict[str, Any]

    # Final trace
    trace: List[Dict[str, Any]]