"""Did Ollama's default context window truncate any generation or judge prompt in the original run?
Rebuilds the exact generation prompt for every retrieving answer in results/detailed.json, asks Ollama for
prompt_eval_count with (a) default options and (b) num_ctx=8192, and compares. If (a) is clipped at a round number
(e.g. 2048 or 4096) while (b) is larger, that prompt was silently truncated in the original run.
Usage (from graph_rag_project/, Ollama running):  python scripts/check_num_ctx.py"""
import json, sys, requests
sys.path.insert(0, ".")
from config import OLLAMA_GENERATE_URL, OLLAMA_MODEL, RESULTS_DIR
from src.generator import RAG_PROMPT, _format_chunks, _format_triples


def ptoks(prompt, options):
    o = dict(options); o["num_predict"] = 1
    r = requests.post(OLLAMA_GENERATE_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": o}, timeout=300)
    return r.json().get("prompt_eval_count", -1)


det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
clipped = 0; rows = []
for d in det:
    if d["system"] == "no_rag":
        continue
    c = d["context"]
    block = "\n\n".join(b for b in (_format_chunks(c["chunks"]), _format_triples(c["triples"])) if b)
    p = RAG_PROMPT.format(context_block=block, question=d["question"])
    a, b = ptoks(p, {}), ptoks(p, {"num_ctx": 8192})
    rows.append((d["system"], a, b)); clipped += (b > a)
by = {}
for s, a, b in rows:
    by.setdefault(s, []).append((a, b))
for s, v in by.items():
    print(s, "max default-window tokens:", max(a for a, _ in v), "max full tokens:", max(b for _, b in v),
          "clipped prompts:", sum(b > a for a, b in v), "/", len(v))
print("TOTAL clipped:", clipped, "of", len(rows))
