"""
Retrievers used by the baselines and the proposed system:

  NaiveRetriever      -- vector search only (the "Naive RAG" baseline)
  BM25HybridRetriever -- lexical (BM25) + vector fusion, no graph (a stronger,
                          commonly-used baseline in the RAG literature -- lets
                          the paper claim more than "graph beats the weakest
                          possible baseline")
  GraphRAGRetriever   -- vector search + graph traversal (the proposed system)

All return a context dict that generator.py turns into a prompt.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.vector_store import search as vector_search
from src.llm import embed_texts
from src.graph_store import match_entities, traverse
from config import (
    TOP_K_CHUNKS, TOP_K_ENTITIES, GRAPH_HOPS, MAX_GRAPH_TRIPLES,
    BM25_TOP_K, BM25_VECTOR_WEIGHT,
)


class NaiveRetriever:
    def __init__(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors

    def retrieve(self, question: str) -> dict:
        top_chunks = vector_search(question, self.chunks, self.vectors, top_k=TOP_K_CHUNKS)
        return {"chunks": top_chunks, "triples": []}


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class BM25HybridRetriever:
    """
    Lexical (BM25) + dense-vector fusion baseline, no knowledge graph.
    This is the "hybrid retrieval without graph structure" comparison point:
    if Graph-RAG only beats this by a small margin, the graph traversal isn't
    pulling its weight; if it beats it clearly (especially on multi-hop
    questions), that is a much stronger claim than beating naive vector-only
    RAG alone.
    """

    def __init__(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

    def retrieve(self, question: str) -> dict:
        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(question)), dtype=np.float32)

        query_vec = np.asarray(embed_texts([question])[0], dtype=np.float32)
        vector_scores = self.vectors @ query_vec  # already L2-normalized -> cosine similarity

        fused = (
            BM25_VECTOR_WEIGHT * _min_max_normalize(vector_scores)
            + (1 - BM25_VECTOR_WEIGHT) * _min_max_normalize(bm25_scores)
        )
        top_idx = np.argsort(-fused)[:BM25_TOP_K]

        results = []
        for idx in top_idx:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(fused[idx])
            results.append(chunk)
        return {"chunks": results, "triples": []}


class GraphRAGRetriever:
    def __init__(self, chunks, vectors, graph):
        self.chunks = chunks
        self.vectors = vectors
        self.graph = graph

    def retrieve(self, question: str) -> dict:
        top_chunks = vector_search(question, self.chunks, self.vectors, top_k=TOP_K_CHUNKS)

        # Seed the graph walk from entities mentioned in the question itself,
        # and from entities mentioned in the top retrieved chunks (helps when
        # the question uses different wording than the graph's node names).
        seed_nodes = set(match_entities(self.graph, question, TOP_K_ENTITIES))
        for chunk in top_chunks:
            seed_nodes.update(match_entities(self.graph, chunk["text"], top_k=2))

        triples = traverse(self.graph, list(seed_nodes), hops=GRAPH_HOPS, max_triples=MAX_GRAPH_TRIPLES)
        return {"chunks": top_chunks, "triples": triples}
