"""
Sample a subset of results/detailed.json for human grading, so the paper can
report agreement between the LLM-judge and a human rater instead of relying on
the LLM-judge scores alone (reviewers at Expert Systems with Applications /
Knowledge-Based Systems will ask for this).

Design choice -- blinding: the file you actually grade (human_eval_blind.csv)
does NOT show which system produced each answer, or what the LLM-judge scored
it, because seeing either would bias your grading toward agreeing with the
judge or toward a system you already believe is "supposed" to win. The mapping
back to system + LLM scores is kept in a separate key file
(human_eval_key.csv) that you should not open until after grading is done.

Usage:
    python scripts/sample_for_human_eval.py [--n 60] [--seed 42]

Output:
    results/human_eval_blind.csv  -- grade this one (fill in human_faithfulness,
                                      human_relevance as 1-5 integers)
    results/human_eval_key.csv    -- do not look at this until you're done grading
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from config import RESULTS_DIR


def _context_excerpt(context: dict, max_chars: int = 800) -> str:
    parts = []
    for c in context.get("chunks", []):
        parts.append(f"[{c['source']}] {c['text']}")
    for t in context.get("triples", []):
        parts.append(f"{t['subject']} --{t['relation']}--> {t['object']}")
    text = "\n".join(parts) if parts else "(no context retrieved)"
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60,
                         help="Total number of (question, system) answers to sample for human grading.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(f"{RESULTS_DIR}/detailed.json", "r", encoding="utf-8") as f:
        detailed = json.load(f)

    # Stratify the sample evenly across systems so no system is over/under-represented
    # in the human-validation subsample.
    by_system = defaultdict(list)
    for i, row in enumerate(detailed):
        by_system[row["system"]].append(i)

    rng = random.Random(args.seed)
    systems = sorted(by_system.keys())
    per_system = max(1, args.n // len(systems))

    sampled_idx = []
    for system in systems:
        idxs = by_system[system][:]
        rng.shuffle(idxs)
        sampled_idx.extend(idxs[:per_system])
    rng.shuffle(sampled_idx)

    blind_rows = []
    key_rows = []
    for eval_id, idx in enumerate(sampled_idx):
        row = detailed[idx]
        blind_rows.append({
            "id": eval_id,
            "question": row["question"],
            "context_excerpt": _context_excerpt(row.get("context", {})),
            "answer": row["answer"],
            "human_faithfulness": "",
            "human_relevance": "",
            "notes": "",
        })
        key_rows.append({
            "id": eval_id,
            "system": row["system"],
            "type": row["type"],
            "llm_faithfulness": row["faithfulness"],
            "llm_relevance": row["relevance"],
            "llm_context_precision": row["context_precision"],
        })

    blind_path = f"{RESULTS_DIR}/human_eval_blind.csv"
    key_path = f"{RESULTS_DIR}/human_eval_key.csv"

    with open(blind_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)

    with open(key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"Sampled {len(blind_rows)} answers ({per_system} per system x {len(systems)} systems).")
    print(f"\n1. Open {blind_path} in a spreadsheet.")
    print("2. For each row, read the question + context_excerpt + answer, then fill in:")
    print("     human_faithfulness  (1-5: is every claim in the answer supported by the context?)")
    print("     human_relevance     (1-5: does the answer directly address the question?)")
    print("   Grade blind -- don't peek at human_eval_key.csv until you're done.")
    print(f"3. Once graded, run: python scripts/compute_agreement.py")


if __name__ == "__main__":
    main()
