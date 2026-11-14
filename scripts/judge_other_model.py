"""Run the revised judge with a different model family on the 60-answer fresh sample (reviewer comment A11).
The answers, contexts and inserted sentences are those of the validation round; only the judge model changes.
Usage (from graph_rag_project/):   ollama pull qwen2.5:7b
                                   python3 scripts/judge_other_model.py qwen2.5:7b          (about 15-40 min)
Writes results/judge_other_<tag>_details.json and results/judge_other_<tag>_scores.csv; the key and the llama3 results are not touched."""
import csv, json, os, re, sys
model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
os.environ["OLLAMA_MODEL"] = model            # must be set before config is imported
sys.path.insert(0, ".")
from config import RESULTS_DIR
from src.evaluate_v2 import judge_v2
tag = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
from src.guard import assert_contexts_restored; assert_contexts_restored(det)
rows = list(csv.DictReader(open(f"{RESULTS_DIR}/human_eval_v2_key.csv", encoding="utf-8")))
details, out = {}, []
for r in rows:
    d = det[int(r["orig_index"])]
    res = judge_v2(d["question"], r["answer_as_shown"], d["context"])
    details[r["id"]] = {k: res.get(k) for k in ("status", "claims", "verdicts", "found_fraction", "faithfulness", "max_prompt_tokens")}
    out.append({"id": r["id"], "inserted": r["inserted"], "judge_score": "" if res["faithfulness"] is None else res["faithfulness"], "status": res["status"]})
    print(r["id"], res["status"], res["faithfulness"], flush=True)
json.dump({"model": model, "items": details}, open(f"{RESULTS_DIR}/judge_other_{tag}_details.json", "w", encoding="utf-8"), indent=1)
with open(f"{RESULTS_DIR}/judge_other_{tag}_scores.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("done", tag)
