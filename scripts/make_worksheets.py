"""Manual-audit worksheets requested by the reviewer. Run from graph_rag_project/ :  python scripts/make_worksheets.py
 1. worksheet_crossdoc_triples.csv  - for the 5 cross-document questions, the triples injected into graph_rag;
                                       annotate `bridging_fact_present` (y/n): does any injected triple state the fact
                                       that connects the two fact sheets? (Comment 3c)
 2. worksheet_triple_precision.csv  - 60 random KG edges with their source chunk text; annotate `correct` (1/0).
 3. worksheet_claim_audit.csv       - (needs results/detailed_v2.json from rejudge_v2.py) 40 random answers with the
                                       judge's extracted claims; annotate missing / merged / invalid claims. (Comment 6e)
Also writes questions_flagged.json = questions.json with "cross_document": true on the five questions."""
import csv, json, pickle, random, os, sys
sys.path.insert(0, ".")
from config import RESULTS_DIR, QUESTIONS_PATH, GRAPH_PATH, CHUNKS_PATH
CROSS = (21, 22, 23, 24, 25)
q = json.load(open(QUESTIONS_PATH, encoding="utf-8"))
det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
rows = []
for i in CROSS:
    for d in det:
        if d["question"] == q[i]["question"] and d["system"] == "graph_rag":
            for t in d["context"]["triples"]:
                rows.append({"question_entry": i + 1, "question": q[i]["question"], "subject": t["subject"],
                             "relation": t["relation"], "object": t["object"], "source": t.get("source"),
                             "bridging_fact_present": ""})
with open(f"{RESULTS_DIR}/worksheet_crossdoc_triples.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for i, x in enumerate(q):
    x["cross_document"] = i in CROSS
json.dump(q, open("questions_flagged.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

g = pickle.load(open(GRAPH_PATH, "rb"))
chunks = {c["chunk_id"]: c for c in pickle.load(open(CHUNKS_PATH, "rb"))}
edges = [(u, v, dd) for u, v, dd in g.edges(data=True)]
random.Random(7).shuffle(edges)
with open(f"{RESULTS_DIR}/worksheet_triple_precision.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["subject", "relation", "object", "source", "chunk_text", "correct_1_or_0"])
    for u, v, dd in edges[:60]:
        c = chunks.get(dd.get("chunk_id"), {})
        w.writerow([u, dd.get("relation"), v, dd.get("source"), c.get("text", ""), ""])
p = f"{RESULTS_DIR}/detailed_v2.json"
if os.path.exists(p):
    d2 = [r for r in json.load(open(p, encoding="utf-8")) if r.get("v2_claims")]
    random.Random(11).shuffle(d2)
    with open(f"{RESULTS_DIR}/worksheet_claim_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["system", "question", "answer", "extracted_claims", "missing_claims", "merged_claims", "invalid_claims"])
        for r in d2[:40]:
            w.writerow([r["system"], r["question"], r["answer"], " || ".join(r["v2_claims"]), "", "", ""])
print("worksheets written to", RESULTS_DIR)
