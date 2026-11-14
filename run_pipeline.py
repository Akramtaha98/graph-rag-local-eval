"""
End-to-end CLI for the Graph-RAG experiment.

Usage:
    python run_pipeline.py build     # ingest docs, extract KG, build vector index
    python run_pipeline.py run       # answer questions.json with all 3 systems + score them
    python run_pipeline.py all       # build then run

Outputs land in results/:
    results/detailed.json   -- every question x system x answer x scores
    results/summary.csv     -- mean scores per system (paste straight into your paper's Results table)
"""

import sys
import json
import argparse

import pandas as pd

from config import (
    QUESTIONS_PATH, GRAPH_PATH, CHUNKS_PATH, VECTOR_PATH, RESULTS_DIR,
)
from src.ingest import build_chunks
from src.kg_extraction import extract_triples
from src.graph_store import build_graph, save_graph, load_graph
from src.vector_store import build_index, save_index, load_index
from src.retriever import NaiveRetriever, BM25HybridRetriever, GraphRAGRetriever
from src.generator import answer_no_rag, answer_naive_rag, answer_bm25_hybrid, answer_graph_rag
from src.evaluate import judge


def cmd_build():
    chunks = build_chunks()
    triples = extract_triples(chunks)
    graph = build_graph(triples)
    save_graph(graph)

    vectors = build_index(chunks)
    save_index(chunks, vectors)
    print("[build] Done. Graph and vector index saved to data/index/.")


def cmd_run():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    chunks, vectors = load_index()
    graph = load_graph()

    naive_retriever = NaiveRetriever(chunks, vectors)
    bm25_retriever = BM25HybridRetriever(chunks, vectors)
    graph_retriever = GraphRAGRetriever(chunks, vectors, graph)

    detailed = []
    for item in questions:
        q = item["question"]
        q_type = item.get("type", "single_hop")
        print(f"\n=== Question ({q_type}): {q}")

        for system_name, run_fn in (
            ("no_rag", lambda: answer_no_rag(q)),
            ("naive_rag", lambda: answer_naive_rag(q, naive_retriever)),
            ("bm25_hybrid", lambda: answer_bm25_hybrid(q, bm25_retriever)),
            ("graph_rag", lambda: answer_graph_rag(q, graph_retriever)),
        ):
            result = run_fn()
            scores = judge(q, result["answer"], result["context"])
            print(f"  [{system_name}] faithfulness={scores['faithfulness']} "
                  f"relevance={scores['relevance']} context_precision={scores['context_precision']}")
            detailed.append({
                "question": q,
                "type": q_type,
                "system": system_name,
                "answer": result["answer"],
                "num_chunks": len(result["context"].get("chunks", [])),
                "num_triples": len(result["context"].get("triples", [])),
                "context": result["context"],
                **scores,
            })

    with open(f"{RESULTS_DIR}/detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(detailed)

    # faithfulness/relevance/context_precision are None (-> NaN) for rows where the
    # judge's response could not be parsed after all retries (see src/evaluate.py).
    # .mean() skips NaN automatically, but report the failure count explicitly per
    # system so a skewed failure rate is visible in the results, not silently
    # baked into the average.
    n_failed = df.groupby("system")["faithfulness"].apply(lambda s: s.isna().sum())

    # faithfulness = groundedness (always 1.0 for no_rag by construction, since it
    # never retrieves anything). faithfulness_world_knowledge is only populated for
    # no_rag -- it's the separate "is this factually correct by general knowledge"
    # judgment, reported alongside rather than conflated into one number.
    score_cols = ["faithfulness", "relevance", "context_precision"]
    if "faithfulness_world_knowledge" in df.columns:
        score_cols.append("faithfulness_world_knowledge")

    summary = df.groupby("system")[score_cols].mean().round(2)
    summary["n_questions"] = df.groupby("system").size()
    summary["n_evaluator_failures"] = n_failed
    summary = summary.reset_index()
    summary.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)

    # multi-hop breakdown -- this is the number the paper's "Discussion" section wants
    multi_hop = df[df["type"] == "multi_hop"]
    if not multi_hop.empty:
        multi_summary = multi_hop.groupby("system")[score_cols].mean().round(2)
        multi_summary.to_csv(f"{RESULTS_DIR}/summary_multi_hop.csv")

    print("\n=== Summary (mean scores per system; NaN rows excluded from means) ===")
    print(summary.to_string(index=False))
    if n_failed.sum() > 0:
        print(f"\nWARNING: {int(n_failed.sum())} judge call(s) failed to parse after all retries "
              f"(see justification='EVALUATOR_FAILURE...' in detailed.json). Excluded from means above, "
              f"not counted as 0.0. If a system's failure rate is much higher than the others', report "
              f"that explicitly -- it's a property of the evaluator prompt length, not answer quality.")
    print(f"\nSaved: {RESULTS_DIR}/summary.csv, {RESULTS_DIR}/detailed.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["build", "run", "all"])
    args = parser.parse_args()

    if args.stage in ("build", "all"):
        cmd_build()
    if args.stage in ("run", "all"):
        cmd_run()


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
