import json
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------------------------------------------------------
# Load policy chunks
# ---------------------------------------------------------

CHUNKS_PATH = Path("app/ingestion/policy_chunks.json")


with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)


# ---------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------

documents = [chunk["text"] for chunk in chunks]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)


# ---------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------

embedding_model = None
document_embeddings = None


def load_dense_model():
    global embedding_model, document_embeddings

    if embedding_model is None:
        print("Loading dense embedding model...")
        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        document_embeddings = embedding_model.encode(
            documents,
            normalize_embeddings=True
        )

        print("Dense embedding model loaded.")


# ---------------------------------------------------------
# BM25 Search
# ---------------------------------------------------------

def bm25_search(query, top_k=5):

    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "page": chunks[index]["page"],
                "section": chunks[index]["section"],
                "text": chunks[index]["text"],
                "score": float(scores[index]),
                "retrieval_method": "bm25",
            }
        )

    return results


# ---------------------------------------------------------
# Dense Search
# ---------------------------------------------------------

def dense_search(query, top_k=5):
    load_dense_model()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "page": chunks[index]["page"],
                "section": chunks[index]["section"],
                "text": chunks[index]["text"],
                "score": float(scores[index]),
                "retrieval_method": "dense",
            }
        )

    return results


# ---------------------------------------------------------
# Simple Hybrid Fusion
# ---------------------------------------------------------

def hybrid_search(query, top_k=5):

    bm25_results = bm25_search(query, top_k=top_k)
    dense_results = dense_search(query, top_k=top_k)

    combined = {}

    # Add BM25 results
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["chunk_id"]

        combined.setdefault(
            chunk_id,
            {
                **result,
                "bm25_rank": None,
                "dense_rank": None,
                "hybrid_score": 0.0,
            }
        )

        combined[chunk_id]["bm25_rank"] = rank

        combined[chunk_id]["hybrid_score"] += 1 / (
            60 + rank
        )

    # Add dense results
    for rank, result in enumerate(dense_results, start=1):

        chunk_id = result["chunk_id"]

        combined.setdefault(
            chunk_id,
            {
                **result,
                "bm25_rank": None,
                "dense_rank": None,
                "hybrid_score": 0.0,
            }
        )

        combined[chunk_id]["dense_rank"] = rank

        combined[chunk_id]["hybrid_score"] += 1 / (
            60 + rank
        )

    # Sort using hybrid score
    results = sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# ---------------------------------------------------------
# Test Retrieval
# ---------------------------------------------------------

if __name__ == "__main__":

    query = "What is the waiting period for pre-existing diseases?"

    print("\nHYBRID SEARCH RESULTS")
    print("=" * 70)

    results = hybrid_search(query, top_k=5)

    for result in results:

        print(f"\nChunk ID: {result['chunk_id']}")
        print(f"Page: {result['page']}")
        print(f"Section: {result['section']}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"BM25 Rank: {result['bm25_rank']}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"Text: {result['text'][:500]}")
        print("-" * 70)