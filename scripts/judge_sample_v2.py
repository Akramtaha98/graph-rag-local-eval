"""Judge the sampled answers EXACTLY as shown to raters (including inserted sentences) with judge v2 and fill the key.
Usage (from graph_rag_project/):  python scripts/judge_sample_v2.py"""
import csv, json, sys
sys.path.insert(0, ".")
from config import RESULTS_DIR
from src.evaluate_v2 import judge_v2

det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
from src.guard import assert_contexts_restored; assert_contexts_restored(det)
rows = list(csv.DictReader(open(f"{RESULTS_DIR}/human_eval_v2_key.csv", encoding="utf-8")))
details = {}
for r in rows:
    d = det[int(r["orig_index"])]
    res = judge_v2(d["question"], r["answer_as_shown"], d["context"])
    details[r["id"]] = {k: res.get(k) for k in ("status", "claims", "verdicts", "found_fraction")}
    r["judge_faithfulness_v2"] = "" if res["faithfulness"] is None else res["faithfulness"]
    r["judge_status"] = res["status"]
with open(f"{RESULTS_DIR}/human_eval_v2_key.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(details, open(f"{RESULTS_DIR}/judge_sample_v2_details.json", "w", encoding="utf-8"), indent=1)
print("done")
