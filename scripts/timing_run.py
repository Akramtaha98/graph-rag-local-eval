"""Wall-clock cost on the stated hardware (reviewer comment A10). Nothing in data/ or results/detailed*.json is overwritten.
Measures: (1) one-time costs: chunking, embedding the chunks, extracting triples with the LLM (155 calls, ~25 min; skip with --skip-extraction),
(2) per-query latency for each system, split into retrieval and generation, over all 54 questions (~30 min).
Usage (from graph_rag_project/, Ollama running, laptop plugged in and otherwise idle):  python3 scripts/timing_run.py   -> results/timing.json"""
import argparse, json, statistics as st, sys, time
sys.path.insert(0, ".")
from config import QUESTIONS_PATH, RESULTS_DIR
from src.ingest import build_chunks
from src.kg_extraction import extract_triples
from src.graph_store import build_graph, load_graph
from src.vector_store import build_index, load_index
from src.retriever import NaiveRetriever, BM25HybridRetriever, GraphRAGRetriever
from src.generator import answer_with_context, answer_no_rag

ap = argparse.ArgumentParser(); ap.add_argument("--skip-extraction", action="store_true"); a = ap.parse_args()
out = {}
t = time.perf_counter(); chunks = build_chunks(); out["chunking_s"] = time.perf_counter() - t
t = time.perf_counter(); build_index(chunks[:1]); out["embedding_model_load_and_first_call_s"] = time.perf_counter() - t
t = time.perf_counter(); build_index(chunks); out["embedding_all_chunks_s"] = time.perf_counter() - t; out["n_chunks"] = len(chunks)
if not a.skip_extraction:
    t = time.perf_counter(); tr = extract_triples(chunks); out["triple_extraction_s"] = time.perf_counter() - t
    t = time.perf_counter(); build_graph(tr); out["graph_build_s"] = time.perf_counter() - t; out["n_triples"] = len(tr)
chunks, vectors = load_index(); graph = load_graph()
R = {"naive_rag": NaiveRetriever(chunks, vectors), "bm25_hybrid": BM25HybridRetriever(chunks, vectors), "graph_rag": GraphRAGRetriever(chunks, vectors, graph)}
qs = [q["question"] for q in json.load(open(QUESTIONS_PATH, encoding="utf-8"))]
answer_no_rag(qs[0])  # warm-up so the first timed call does not include model loading
lat = {k: {"retrieval_s": [], "generation_s": []} for k in R}; lat["no_rag"] = {"retrieval_s": [], "generation_s": []}
for q in qs:
    t = time.perf_counter(); answer_no_rag(q); lat["no_rag"]["generation_s"].append(time.perf_counter() - t); lat["no_rag"]["retrieval_s"].append(0.0)
    for s, r in R.items():
        t = time.perf_counter(); ctx = r.retrieve(q); t1 = time.perf_counter(); answer_with_context(q, ctx); t2 = time.perf_counter()
        lat[s]["retrieval_s"].append(t1 - t); lat[s]["generation_s"].append(t2 - t1)
out["per_query"] = {s: {k: {"mean": st.mean(v), "median": st.median(v), "max": max(v)} for k, v in d.items()} for s, d in lat.items()}
out["per_query"]["_total_mean_s"] = {s: st.mean(a + b for a, b in zip(d["retrieval_s"], d["generation_s"])) for s, d in lat.items()}
json.dump(out, open(f"{RESULTS_DIR}/timing.json", "w"), indent=1)
print(json.dumps(out, indent=1))
