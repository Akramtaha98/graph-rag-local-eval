"""
Minimal local vector store: embed all chunks once with sentence-transformers,
keep them as a numpy matrix, and do cosine-similarity search in memory.
No FAISS/Chroma dependency needed for the corpus sizes a laptop experiment
will use (thousands of chunks is still instant with numpy).
"""

import pickle
import numpy as np

from src.llm import embed_texts
from config import CHUNKS_PATH, VECTOR_PATH, TOP_K_CHUNKS


def build_index(chunks: list[dict]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)
    vectors = np.asarray(vectors, dtype=np.float32)
    print(f"[vector_store] Embedded {len(texts)} chunks -> shape {vectors.shape}.")
    return vectors


def save_index(chunks: list[dict], vectors: np.ndarray,
                chunks_path: str = CHUNKS_PATH, vector_path: str = VECTOR_PATH) -> None:
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    np.save(vector_path, vectors)


def load_index(chunks_path: str = CHUNKS_PATH, vector_path: str = VECTOR_PATH):
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    vectors = np.load(vector_path)
    return chunks, vectors


def search(query: str, chunks: list[dict], vectors: np.ndarray, top_k: int = TOP_K_CHUNKS) -> list[dict]:
    """Return the top_k chunks most similar to the query (cosine similarity)."""
    query_vec = embed_texts([query])[0]
    query_vec = np.asarray(query_vec, dtype=np.float32)
    # vectors are already L2-normalized (normalize_embeddings=True in embed_texts)
    scores = vectors @ query_vec
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_idx:
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[idx])
        results.append(chunk)
    return results
