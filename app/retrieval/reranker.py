from sentence_transformers import CrossEncoder


# Load reranking model
reranker_model = None


def load_reranker_model():
    global reranker_model

    if reranker_model is None:
        print("Loading reranker model...")
        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        print("Reranker model loaded.")


def rerank_results(query, results, top_k=5):
    load_reranker_model()
    
    """
    Rerank hybrid retrieval results using a cross-encoder.
    """

    if not results:
        return []

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = reranker_model.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):

        updated_result = result.copy()

        updated_result["reranker_score"] = float(score)

        reranked.append(updated_result)

    # Highest reranker score first
    reranked.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked[:top_k]


if __name__ == "__main__":

    from hybrid_retriever import hybrid_search

    query = "What is the waiting period for pre-existing diseases?"

    hybrid_results = hybrid_search(
        query,
        top_k=10
    )

    reranked_results = rerank_results(
        query,
        hybrid_results,
        top_k=5
    )

    print("\nRERANKED RESULTS")
    print("=" * 70)

    for result in reranked_results:

        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Section: {result['section']}")
        print(
            f"Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )
        print(
            f"BM25 Rank: "
            f"{result['bm25_rank']}"
        )
        print(
            f"Dense Rank: "
            f"{result['dense_rank']}"
        )
        print(
            f"Text: "
            f"{result['text'][:400]}"
        )
        print("-" * 70)