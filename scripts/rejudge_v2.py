"""Re-judge every answer in results/detailed.json with judge v2 (full context, per-claim verification).
Writes results/detailed_v2.json. Does NOT overwrite detailed.json. ~4-6 judge calls per answer.
Usage (from graph_rag_project/):  python scripts/rejudge_v2.py
Check at the end: max_prompt_tokens must be well below NUM_CTX (8192)."""
import json
import sys
sys.path.insert(0, ".")
from config import RESULTS_DIR
from src.evaluate_v2 import judge_v2


def main():
    det = json.load(open(f"{RESULTS_DIR}/detailed.json", encoding="utf-8"))
    from src.guard import assert_contexts_restored; assert_contexts_restored(det)
    out = []
    for i, row in enumerate(det):
        res = judge_v2(row["question"], row["answer"], row.get("context", {"chunks": [], "triples": []}))
        new = dict(row)
        new.update({f"v2_{k}": v for k, v in res.items()})
        out.append(new)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(det)}")
            json.dump(out, open(f"{RESULTS_DIR}/detailed_v2.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(out, open(f"{RESULTS_DIR}/detailed_v2.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    from collections import Counter
    print("status by system:", {s: dict(Counter(r["v2_status"] for r in out if r["system"] == s))
                                for s in sorted({r["system"] for r in out})})
    print("max prompt tokens seen:", max(r["v2_max_prompt_tokens"] for r in out))


if __name__ == "__main__":
    main()
