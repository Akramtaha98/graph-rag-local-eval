"""Maximum prompt length in tokens for each prompt type, with Ollama's prompt cache defeated (reviewer comment A6).
Ollama's prompt_eval_count can omit tokens that were reused from the previous request, so every prompt here starts
with a unique nonce line; the nonce's own token cost is measured once and subtracted.
Prompt types: generation, claim extraction, claim verification (context + longest stored claim), relevance (context + answer).
Usage (from graph_rag_project/, Ollama running):  python3 scripts/token_audit.py   -> results/token_audit.json"""
import json, sys, uuid
import requests
sys.path.insert(0, ".")
from config import OLLAMA_GENERATE_URL, OLLAMA_MODEL, RESULTS_DIR
from src.generator import RAG_PROMPT, _format_chunks, _format_triples
from src.evaluate import _context_to_text, RELEVANCE_CONTEXT_PROMPT
from src.evaluate_v2 import CLAIM_EXTRACTION_PROMPT, CLAIM_CHECK_PROMPT
NUM_CTX = 8192


def ptoks(prompt):
    nonce = f"[request {uuid.uuid4()}]\n"
    r = requests.post(OLLAMA_GENERATE_URL, json={"model": OLLAMA_MODEL, "prompt": nonce + prompt, "stream": False,
                                                 "options": {"num_ctx": NUM_CTX, "num_predict": 1, "temperature": 0}}, timeout=600)
    return int(r.json().get("prompt_eval_count", -1))


overhead = min(ptoks("") for _ in range(3))
det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
v2 = json.load(open(f"{RESULTS_DIR}/detailed_v2.json", encoding="utf-8"))
res = {}
for d, e in zip(det, v2):
    s = d["system"]
    if s == "no_rag":
        continue
    c = d["context"]
    block = "\n\n".join(b for b in (_format_chunks(c["chunks"]), _format_triples(c["triples"])) if b)
    ctx = _context_to_text(c)
    claims = json.loads(e["v2_claims"]) if isinstance(e["v2_claims"], str) else (e["v2_claims"] or [])
    longest = max(claims, key=len) if claims else "(none)"
    row = {"generation": ptoks(RAG_PROMPT.format(context_block=block, question=d["question"])) - overhead,
           "extraction": ptoks(CLAIM_EXTRACTION_PROMPT.format(answer=d["answer"])) - overhead,
           "verification": ptoks(CLAIM_CHECK_PROMPT.format(context=ctx, claim=longest)) - overhead,
           "relevance": ptoks(RELEVANCE_CONTEXT_PROMPT.format(question=d["question"], context=ctx, answer=d["answer"])) - overhead,
           "context_chars": len(ctx)}
    for k, v in row.items():
        res.setdefault(s, {}).setdefault(k, []).append(v)
summary = {s: {k: max(v) for k, v in d.items()} for s, d in res.items()}
allmax = {k: max(summary[s][k] for s in summary) for k in next(iter(summary.values()))}
out = {"nonce_overhead_tokens": overhead, "num_ctx": NUM_CTX, "max_by_system": summary, "max_overall": allmax,
       "any_prompt_at_or_above_window": any(v >= NUM_CTX for d in summary.values() for v in d.values())}
json.dump(out, open(f"{RESULTS_DIR}/token_audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
