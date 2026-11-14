"""Classify refusals in results/detailed_v2.json and print the table used in Section 4.1.
Usage (from graph_rag_project/): python scripts/refusal_audit.py"""
import json, sys
sys.path.insert(0, ".")
from src.refusal import refusal_type
d = json.load(open("results/detailed_v2.json", encoding="utf-8"))
for i, r in enumerate(d):
    if r["system"] != "no_rag":
        t = refusal_type(r["answer"])
        if t != "none":
            print(i, r["system"], t, "revised faithfulness:", r["v2_faithfulness"], "relevance:", r["v2_relevance"], "|", r["question"][:60])
