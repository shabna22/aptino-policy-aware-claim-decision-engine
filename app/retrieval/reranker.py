import re


# ---------------------------------------------------------
# Lightweight Policy Evidence Reranker
# ---------------------------------------------------------

def _tokenize(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def _rerank_score(query, text):
    """
    Lightweight lexical relevance score.

    Combines:
    - query term overlap
    - exact phrase matches
    - policy terminology overlap
    """

    query_tokens = _tokenize(query)
    text_tokens = _tokenize(text)

    if not query_tokens or not text_tokens:
        return 0.0

    # Token overlap
    overlap = len(query_tokens & text_tokens)

    # Normalize by query size
    overlap_score = overlap / len(query_tokens)

    # Exact phrase bonus
    query_lower = query.lower()
    text_lower = text.lower()

    phrase_bonus = 0.0

    if query_lower in text_lower:
        phrase_bonus = 0.5

    # Important policy terms receive additional relevance
    policy_terms = {
        "waiting",
        "period",
        "exclusion",
        "coverage",
        "hospitalization",
        "domiciliary",
        "day care",
        "pre-existing",
        "preexisting",
        "experimental",
        "cosmetic",
        "limit",
        "sub-limit",
        "definition",
        "portability",
    }

    query_policy_terms = query_tokens & policy_terms
    text_policy_terms = text_tokens & policy_terms

    policy_overlap = len(
        query_policy_terms & text_policy_terms
    )

    policy_bonus = (
        policy_overlap / max(len(query_policy_terms), 1)
    ) * 0.3

    return (
        overlap_score
        + phrase_bonus
        + policy_bonus
    )


def rerank_results(query, results, top_k=5):

    reranked = []

    for result in results:

        score = _rerank_score(
            query,
            result["text"]
        )

        updated_result = {
            **result,
            "reranker_score": float(score),
        }

        reranked.append(updated_result)

    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked[:top_k]