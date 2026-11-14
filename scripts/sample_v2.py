"""Fresh validation sample for human grading (reviewer Comments 2 and 5).
 * New seed, drawn ONLY from retrieving systems (no_rag agrees by construction and inflates agreement).
 * Excludes every (question, system) pair used in the first validation round.
 * Raters see the FULL context exactly as the judge sees it, INCLUDING graph triples (no 800-character cut).
 * A known number of answers get an inserted unsupported sentence (taken from a chunk of a DIFFERENT source
   document), so the sample is guaranteed to contain unfaithful answers. Which rows were modified is in the key file only.
 * Writes one blind CSV per rater (identical content, separate files) and one key file.
Usage (from graph_rag_project/):
    python scripts/sample_v2.py --n 60 --inject 12 --seed 2026 --raters 3
Raters must be outside the author team, must not see the key or each other's files, and fill human_faithfulness / human_relevance."""
import argparse, csv, json, random, re, sys, pickle
from collections import defaultdict
sys.path.insert(0, ".")
from config import RESULTS_DIR, CHUNKS_PATH


def full_context(ctx):
    parts = [f"[{c['source']}] {c['text']}" for c in ctx.get("chunks", [])]
    if ctx.get("triples"):
        parts.append("Knowledge graph relations:")
        parts += [f"{t['subject']} --{t['relation']}--> {t['object']}" for t in ctx["triples"]]
    return "\n".join(parts)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if 60 < len(s.strip()) < 260]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--inject", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--raters", type=int, default=3)
    ap.add_argument("--results", default="detailed.json")
    a = ap.parse_args()
    det = json.load(open(f"{RESULTS_DIR}/{a.results}", encoding="utf-8"))
    used = {(r["question"], k["system"]) for r, k in zip(csv.DictReader(open(f"{RESULTS_DIR}/human_eval_blind.csv", encoding="utf-8")),
                                                            csv.DictReader(open(f"{RESULTS_DIR}/human_eval_key.csv", encoding="utf-8")))}
    chunks = pickle.load(open(CHUNKS_PATH, "rb"))
    rng = random.Random(a.seed)
    by_sys = defaultdict(list)
    for i, r in enumerate(det):
        if r["system"] != "no_rag" and (r["question"], r["system"]) not in used and r["answer"].strip():
            by_sys[r["system"]].append(i)
    systems = sorted(by_sys)
    per = a.n // len(systems)
    picked = []
    for s in systems:
        rng.shuffle(by_sys[s]); picked += by_sys[s][:per]
    rng.shuffle(picked)
    inject_ids = set(rng.sample(range(len(picked)), min(a.inject, len(picked))))
    blind, key = [], []
    for eid, idx in enumerate(picked):
        r = det[idx]
        ans, inserted, ins_text = r["answer"], 0, ""
        if eid in inject_ids:
            srcs = {c["source"] for c in r["context"]["chunks"]}
            pool = [s for c in chunks if c["source"] not in srcs for s in sentences(c["text"])]
            if pool:
                ins_text = rng.choice(pool)
                ans = ans.rstrip() + " " + ins_text
                inserted = 1
        blind.append({"id": eid, "question": r["question"], "context": full_context(r["context"]), "answer": ans,
                      "human_faithfulness": "", "human_relevance": "", "notes": ""})
        key.append({"id": eid, "orig_index": idx, "system": r["system"], "type": r["type"], "inserted": inserted,
                    "inserted_text": ins_text, "answer_as_shown": ans, "judge_faithfulness_v2": "", "judge_status": ""})
    for k in range(1, a.raters + 1):
        with open(f"{RESULTS_DIR}/human_eval_v2_rater{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(blind[0])); w.writeheader(); w.writerows(blind)
    with open(f"{RESULTS_DIR}/human_eval_v2_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    print(f"{len(blind)} items; {sum(k['inserted'] for k in key)} with an inserted unsupported sentence; "
          f"{a.raters} rater files + key written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
