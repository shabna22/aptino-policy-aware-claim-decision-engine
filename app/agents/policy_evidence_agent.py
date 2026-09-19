from app.agents.state import ClaimState
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank_results


def policy_evidence_agent(state: ClaimState) -> ClaimState:
    """
    Retrieve policy evidence using multiple targeted queries
    derived from the Case Analysis Agent.
    """

    case_analysis = state["case_analysis"]

    decision_dimensions = case_analysis.get(
        "decision_dimensions",
        []
    )

    investigation_plan = case_analysis.get(
        "investigation_plan",
        []
    )

    # Create separate retrieval queries instead of
    # combining everything into one large query.
    queries = []

    for item in decision_dimensions:
        queries.append(str(item))

    for item in investigation_plan:
        queries.append(str(item))

    # Always include the claim itself as a fallback/context query.
    claim = state["claim"]

    queries.append(
        f"{claim.get('claim_type', '')} "
        f"{claim.get('treatment', '')} "
        f"{claim.get('hospitalization_type', '')}"
    )

    # Remove empty and duplicate queries
    queries = list(
        dict.fromkeys(
            q.strip()
            for q in queries
            if q.strip()
        )
    )

    all_results = {}

    # Retrieve evidence for every targeted query
    for query in queries:

        hybrid_results = hybrid_search(
            query,
            top_k=10
        )

        reranked_results = rerank_results(
            query,
            hybrid_results,
            top_k=5
        )

        for result in reranked_results:

            chunk_id = result["chunk_id"]

            # Keep the strongest score if the same
            # chunk appears for multiple queries.
            if (
                chunk_id not in all_results
                or result["reranker_score"]
                > all_results[chunk_id]["reranker_score"]
            ):

                result = result.copy()

                result["retrieval_query"] = query

                all_results[chunk_id] = result

    # Sort all unique evidence by reranker score
    final_results = sorted(
        all_results.values(),
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    # Keep the strongest evidence
    final_results = final_results[:10]

    state["retrieved_evidence"] = final_results

    state.setdefault("trace", []).append(
        {
            "agent": "Policy Evidence Agent",
            "action": (
                "Retrieved policy evidence using multiple "
                "targeted hybrid retrieval queries followed "
                "by cross-encoder reranking."
            ),
            "query_count": len(queries),
            "evidence_count": len(final_results)
        }
    )

    return state