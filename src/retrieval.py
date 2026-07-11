"""Hybrid (vector + keyword) retrieval over the indexed chunks.

Combines Chroma's semantic search with BM25 keyword search via reciprocal
rank fusion (RRF): each chunk's final score is the sum of 1/(k + rank) from
each method it appears in. This means a chunk ranked highly by *either*
method surfaces near the top, without needing to calibrate the two
methods' raw scores against each other (which aren't on the same scale --
cosine similarity vs. BM25 score).
"""
import pickle
from pathlib import Path
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
BM25_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "bm25.pkl"
COLLECTION_NAME = "elden_ring_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RRF_K = 60  # standard RRF constant; de-emphasizes rank differences beyond the top few

# If the query names a specific page (e.g. "Ancient Death Rancor patch"),
# strongly prefer that page's own chunks over other pages that happen to
# score better on the query's other terms. Without this, a query mixing an
# entity name with a topic keyword (e.g. "<item> patch changes") routinely
# loses to *other* pages' patch-note-heavy chunks, even though the
# entity's own (less keyword-dense) chunk is the actually-correct answer --
# confirmed empirically: "Ancient Death Rancor patch" ranked that page's
# own Description chunk (which has the patch note) 13th, crowded out by
# unrelated bosses/items that merely score higher on "patch" alone.
# 0.05 comfortably exceeds the largest possible RRF score (two rank-1 hits
# ~= 0.033), so any title match outranks any non-match, while chunks
# *within* the matched title still sort by their normal RRF score.
TITLE_BOOST = 0.05


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # explicit device="cpu": on HF Spaces' ZeroGPU hardware, the `spaces`
    # package globally patches torch's CUDA init to raise unless it's
    # called from inside an @spaces.GPU-wrapped function (ZeroGPU only
    # grants GPU access transiently, scoped to those calls). Without this,
    # SentenceTransformer's default device auto-detection tries CUDA
    # (since ZeroGPU hardware nominally has one) and trips that guard --
    # confirmed via a live traceback: "RuntimeError: Low-level CUDA init
    # ... did not intercept a CUDA operation". This app never needs a GPU
    # at all (embeddings are cheap enough for CPU; the actual LLM calls go
    # to Groq's remote API), so forcing CPU avoids the CUDA path entirely.
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_bm25():
    with BM25_PATH.open("rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunks"]


def _vector_search(query: str, top_n: int) -> list[str]:
    """Returns chunk ids ranked by semantic similarity."""
    model = _get_model()
    collection = _get_collection()
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_n)
    return results["ids"][0]


def _keyword_search(query: str, top_n: int) -> list[str]:
    """Returns chunk ids ranked by BM25 score."""
    bm25, chunks = _get_bm25()
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i]["id"] for i in ranked[:top_n] if scores[i] > 0]


def search(query: str, k: int = 5, candidate_pool: int = 20) -> list[dict]:
    """Hybrid search: fuse vector + keyword rankings, return top-k chunks
    with full text + metadata for citation."""
    vector_ids = _vector_search(query, candidate_pool)
    keyword_ids = _keyword_search(query, candidate_pool)

    rrf_scores: dict[str, float] = {}
    for rank, cid in enumerate(vector_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
    for rank, cid in enumerate(keyword_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)

    # fetch metadata for the whole candidate pool up front so title-boost
    # can be applied before truncating to k
    candidate_ids = list(rrf_scores.keys())
    collection = _get_collection()
    fetched = collection.get(ids=candidate_ids, include=["documents", "metadatas"])
    id_to_result = {
        cid: {"id": cid, "text": doc, **meta}
        for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])
    }

    query_lower = query.lower()
    for cid, result in id_to_result.items():
        if result["title"].lower() in query_lower:
            rrf_scores[cid] += TITLE_BOOST

    top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:k]
    return [id_to_result[cid] for cid in top_ids if cid in id_to_result]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "what is the FP cost of Ancient Death Rancor"
    print(f"Query: {query}\n")
    for r in search(query, k=5):
        print(f"[{r['section']}] {r['title']} ({r['category']}) — score chunk")
        print(f"  {r['text'][:200]}")
        print(f"  {r['url']}\n")
