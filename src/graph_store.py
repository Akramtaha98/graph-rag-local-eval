"""
NetworkX-based knowledge graph store. No external database required --
this is the "Neo4j / NetworkX" storage layer from the blueprint, implemented
with NetworkX so the whole project runs with zero infrastructure setup.

If you later want to scale to Neo4j, the triples produced by kg_extraction.py
are already in the right shape to load via `UNWIND $triples AS t MERGE ...`.
"""

import pickle
import networkx as nx

from config import GRAPH_PATH, MIN_ENTITY_MATCH_RATIO, MAX_ENTITY_NODE_WORDS


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for t in triples:
        subj, rel, obj = t["subject"], t["relation"], t["object"]
        graph.add_node(subj)
        graph.add_node(obj)
        graph.add_edge(
            subj, obj,
            relation=rel,
            chunk_id=t.get("chunk_id"),
            source=t.get("source"),
        )
    print(f"[graph_store] Built graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    return graph


def save_graph(graph: nx.MultiDiGraph, path: str = GRAPH_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path: str = GRAPH_PATH) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def match_entities(graph: nx.MultiDiGraph, query: str, top_k: int) -> list[str]:
    """
    Simple lexical matching: score each node by the fraction of its lowercase
    tokens that appear in the query. Only nodes clearing MIN_ENTITY_MATCH_RATIO
    are kept, and overly long nodes (usually mis-extracted sentence fragments
    rather than real entities) are skipped -- both filters materially reduced
    irrelevant/noisy triples being pulled into the context during testing.
    Good enough for a first working system; swap in embedding-based entity
    linking later if you need higher recall on paraphrased entity mentions.
    """
    query_lower = query.lower()
    scored = []
    for node in graph.nodes:
        node_words = node.split()
        if len(node_words) > MAX_ENTITY_NODE_WORDS:
            continue
        node_tokens = [tok for tok in node.lower().split() if len(tok) > 2]
        if not node_tokens:
            continue
        hits = sum(1 for tok in node_tokens if tok in query_lower)
        ratio = hits / len(node_tokens)
        if ratio >= MIN_ENTITY_MATCH_RATIO:
            scored.append((ratio, node))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [node for _, node in scored[:top_k]]


def traverse(graph: nx.MultiDiGraph, seed_nodes: list[str], hops: int, max_triples: int) -> list[dict]:
    """
    Breadth-first traversal from seed_nodes up to `hops` hops (following
    edges in both directions), returning the underlying triples for
    multi-hop context.
    """
    visited_nodes = set(seed_nodes)
    frontier = list(seed_nodes)
    collected = []

    for _ in range(hops):
        next_frontier = []
        for node in frontier:
            if node not in graph:
                continue
            for _, target, data in graph.out_edges(node, data=True):
                collected.append({"subject": node, "relation": data.get("relation"), "object": target,
                                   "source": data.get("source")})
                if target not in visited_nodes:
                    visited_nodes.add(target)
                    next_frontier.append(target)
            for source, _, data in graph.in_edges(node, data=True):
                collected.append({"subject": source, "relation": data.get("relation"), "object": node,
                                   "source": data.get("source")})
                if source not in visited_nodes:
                    visited_nodes.add(source)
                    next_frontier.append(source)
        frontier = next_frontier
        if len(collected) >= max_triples:
            break

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for triple in collected:
        key = (triple["subject"], triple["relation"], triple["object"])
        if key not in seen:
            seen.add(key)
            deduped.append(triple)
    return deduped[:max_triples]
