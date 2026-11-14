"""
Compute agreement between your human grades (results/human_eval_blind.csv,
filled in) and the LLM-judge scores (results/human_eval_key.csv), produced by
scripts/sample_for_human_eval.py.

This is the number your paper reports to justify using an LLM-as-judge instead
of full human annotation, e.g.:
    "LLM-judge faithfulness scores correlated with human ratings at r=0.82
    (Pearson) on a blinded 60-answer subsample; 78% of scores were within
    1 point of the human rating."

Usage:
    python scripts/compute_agreement.py
"""

import csv
import sys

sys.path.insert(0, ".")
from config import RESULTS_DIR


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    return cov / denom if denom > 1e-12 else float("nan")


def _load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    blind_path = f"{RESULTS_DIR}/human_eval_blind.csv"
    key_path = f"{RESULTS_DIR}/human_eval_key.csv"

    blind_rows = {r["id"]: r for r in _load_csv(blind_path)}
    key_rows = {r["id"]: r for r in _load_csv(key_path)}

    missing = [i for i, r in blind_rows.items() if not r["human_faithfulness"].strip() or not r["human_relevance"].strip()]
    if missing:
        print(f"WARNING: {len(missing)} row(s) still have blank human_faithfulness/human_relevance -- "
              f"finish grading {blind_path} before running this. Continuing with the rows that are filled in.")

    faith_human, faith_llm = [], []
    rel_human, rel_llm = [], []
    per_system = {}

    n_evaluator_failures_in_sample = 0
    for eval_id, brow in blind_rows.items():
        if not brow["human_faithfulness"].strip() or not brow["human_relevance"].strip():
            continue
        krow = key_rows[eval_id]
        if not krow["llm_faithfulness"].strip() or not krow["llm_relevance"].strip():
            # This row's LLM-judge call failed after all retries (see
            # src/evaluate.py / results/judge_failures.log) -- there's no LLM
            # score to compare against, so skip it rather than crash. Report
            # how many were skipped so it's visible, not silently dropped.
            n_evaluator_failures_in_sample += 1
            continue
        hf, hr = float(brow["human_faithfulness"]), float(brow["human_relevance"])
        lf, lr = float(krow["llm_faithfulness"]), float(krow["llm_relevance"])

        faith_human.append(hf); faith_llm.append(lf)
        rel_human.append(hr); rel_llm.append(lr)

        system = krow["system"]
        per_system.setdefault(system, {"faith_human": [], "faith_llm": [], "rel_human": [], "rel_llm": []})
        per_system[system]["faith_human"].append(hf)
        per_system[system]["faith_llm"].append(lf)
        per_system[system]["rel_human"].append(hr)
        per_system[system]["rel_llm"].append(lr)

    n = len(faith_human)
    if n == 0:
        print("No graded rows found yet -- fill in human_eval_blind.csv first.")
        return

    within_1_faith = sum(1 for h, l in zip(faith_human, faith_llm) if abs(h - l) <= 1) / n
    within_1_rel = sum(1 for h, l in zip(rel_human, rel_llm) if abs(h - l) <= 1) / n

    print(f"=== Human vs. LLM-judge agreement (n={n} graded answers"
          f"{f', {n_evaluator_failures_in_sample} skipped due to evaluator failures' if n_evaluator_failures_in_sample else ''}) ===\n")
    print(f"Faithfulness: Pearson r = {_pearson(faith_human, faith_llm):.3f}, "
          f"{within_1_faith*100:.0f}% within 1 point")
    print(f"Relevance:    Pearson r = {_pearson(rel_human, rel_llm):.3f}, "
          f"{within_1_rel*100:.0f}% within 1 point")

    print("\n=== Per-system breakdown ===")
    for system, d in sorted(per_system.items()):
        ns = len(d["faith_human"])
        print(f"  {system} (n={ns}): "
              f"faithfulness r={_pearson(d['faith_human'], d['faith_llm']):.3f}, "
              f"relevance r={_pearson(d['rel_human'], d['rel_llm']):.3f}")

    report_path = f"{RESULTS_DIR}/human_validation_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "pearson_r", "pct_within_1_point", "n"])
        writer.writerow(["faithfulness", round(_pearson(faith_human, faith_llm), 3), round(within_1_faith, 3), n])
        writer.writerow(["relevance", round(_pearson(rel_human, rel_llm), 3), round(within_1_rel, 3), n])
    print(f"\nSaved: {report_path} -- paste these numbers into your paper's Methods/Evaluation section.")


if __name__ == "__main__":
    main()
