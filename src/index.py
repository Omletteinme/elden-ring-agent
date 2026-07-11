"""Build the vector store + keyword index from data/chunks/chunks.jsonl.

Two indexes over the same chunks, combined at query time (see
retrieval.py):
  - Chroma (vector, sentence-transformers embeddings) -- catches semantic
    similarity ("how do I resist frost" ~ "immunity to cold").
  - BM25 (keyword) -- catches exact-name lookups ("Adula's Moonblade")
    that embeddings alone are often loose on, since game item names are
    out-of-distribution proper nouns.

Run this after chunking.py, and again any time the chunks change.
"""
import json
import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "chunks.jsonl"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
BM25_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks" / "bm25.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "elden_ring_chunks"
EMBED_BATCH_SIZE = 64


def load_chunks() -> list[dict]:
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]


def build_vector_index(chunks: list[dict]) -> None:
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    # explicit device="cpu" -- see the matching comment in retrieval.py's
    # _get_model() for why (ZeroGPU's global CUDA-init guard on HF Spaces)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"title": c["title"], "category": c["category"], "url": c["url"], "section": c["section"]} for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)

    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)
    print(f"Vector index built: {collection.count()} chunks in {CHROMA_DIR}")


def build_bm25_index(chunks: list[dict]) -> None:
    print("Building BM25 keyword index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index built: {BM25_PATH}")


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")
    build_vector_index(chunks)
    print()
    build_bm25_index(chunks)


if __name__ == "__main__":
    main()
