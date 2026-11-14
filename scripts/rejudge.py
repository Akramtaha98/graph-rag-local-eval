"""
Re-score the answers already saved in results/detailed.json with an updated
judge prompt, WITHOUT regenerating the answers themselves.

Why this exists: the human-validation subsample (see scripts/compute_agreement.py)
found essentially no correlation between the LLM-judge's faithfulness scores and
a blinded human rater (Pearson r = -0.18 overall, as low as r = -0.43 for some
systems). Digging into the human grader's notes showed a specific, systematic
cause: the original judge prompt said "if there is no context, judge faithfulness
against general world knowledge instead" -- and the judge model applied that
leniency rule even when CONTEXT was non-empty but simply didn't contain the fact
needed, effectively grading factual correctness instead of groundedness. 15 of the
60 human-graded rows were flagged by the human rater as "answer is correct but not
actually supported by the retrieved context" and the judge scored 12 of those 15
as faithfulness=5 anyway.

src/evaluate.py's JUDGE_PROMPT has been rewritten to make this distinction
explicit. This script re-runs judging only (not generation) over the existing
answers/contexts in detailed.json, which is much cheaper than a full rerun --
it's ~216 judge calls instead of ~432 generation+judge calls.

Usage:
    python scripts/rejudge.py
"""

import json
import sys

import pandas as pd

sys.path.insert(0, ".")
from config import RESULTS_DIR
from src.evaluate import judge


def main():
    with open(f"{RESULTS_DIR}/detailed.json", "r", encoding="utf-8") as f:
        detailed = json.load(f)

    print(f"Re-judging {len(detailed)} existing answers with the corrected faithfulness prompt...")
    for i, row in enumerate(detailed):
        context = row.get("context", {"chunks": [], "triples": []})
        scores = judge(row["question"], row["answer"], context)
        row["faithfulness"] = scores["faithfulness"]
        row["faithfulness_world_knowledge"] = scores.get("faithfulness_world_knowledge")
        row["relevance"] = scores["relevance"]
        row["context_precision"] = scores["context_precision"]
        row["justification"] = scores["justification"]
        if (i + 1) % 20 == 0 or i == len(detailed) - 1:
            print(f"  {i + 1}/{len(detailed)} re-judged")

    with open(f"{RESULTS_DIR}/detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(detailed)
    n_failed = df.groupby("system")["faithfulness"].apply(lambda s: s.isna().sum())
    # faithfulness = groundedness (always 1.0 for no_rag by construction).
    # faithfulness_world_knowledge is the separate "factually correct by general
    # knowledge" judgment, only populated for no_rag -- reported alongside, not
    # merged into the main faithfulness number.
    score_cols = ["faithfulness", "relevance", "context_precision"]
    if "faithfulness_world_knowledge" in df.columns:
        score_cols.append("faithfulness_world_knowledge")
    summary = df.groupby("system")[score_cols].mean().round(2)
    summary["n_questions"] = df.groupby("system").size()
    summary["n_evaluator_failures"] = n_failed
    summary = summary.reset_index()
    summary.to_csv(f"{RESULTS_DIR}/summary.csv", index=False)

    multi_hop = df[df["type"] == "multi_hop"]
    if not multi_hop.empty:
        multi_summary = multi_hop.groupby("system")[score_cols].mean().round(2)
        multi_summary.to_csv(f"{RESULTS_DIR}/summary_multi_hop.csv")

    print("\n=== Re-judged summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR}/detailed.json, {RESULTS_DIR}/summary.csv, {RESULTS_DIR}/summary_multi_hop.csv")

    # Sanity check: a judge that is actually reading each answer should produce a
    # spread of scores, not pile up on one value. This has failed twice before in
    # two different ways: (1) every row came back exactly faithfulness=4,
    # relevance=5 -- literally the example numbers copied from the JSON template;
    # (2) less obviously, 96% of rows came back faithfulness=4.0 with only rare
    # exceptions -- technically >1 distinct value, but still a collapse. So this
    # checks the concentration of the single most common value, not just whether
    # there's more than one.
    warnings = []
    for col in ("faithfulness", "relevance", "context_precision"):
        valid = df[col].dropna()
        if valid.empty:
            continue
        top_value, top_count = valid.value_counts().idxmax(), valid.value_counts().max()
        top_share = top_count / len(valid)
        if top_share >= 0.85:
            warnings.append(
                f"{col}: {top_share*100:.0f}% of rows are exactly {top_value} "
                f"({top_count}/{len(valid)}) -- looks collapsed, not real per-answer judgment."
            )

    if warnings:
        print("\n*** WARNING: possible evaluator collapse detected. Do NOT trust this run or "
              "grade a human sample against it yet: ***")
        for w in warnings:
            print("  -", w)
    else:
        for col in ("faithfulness", "relevance", "context_precision"):
            valid = df[col].dropna()
            if valid.empty:
                continue
            n_unique = valid.nunique()
            print(f"Sanity check ({col}): {n_unique} distinct values across {len(valid)} rows, "
                  f"largest single value is {valid.value_counts().max()/len(valid)*100:.0f}% of rows.")

    print("\nNext: re-run scripts/sample_for_human_eval.py + grade a fresh subsample + "
          "scripts/compute_agreement.py to check whether the corrected prompt actually "
          "improved human/judge agreement.")


if __name__ == "__main__":
    main()
