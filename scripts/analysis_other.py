"""Compare a second judge model with the first judge and the raters on the fresh sample (reviewer comment A11). No model is called.
Usage: python3 scripts/analysis_other.py results/judge_other_<tag>_details.json [results_dir]"""
import csv, json, math, os, sys
import numpy as np
from scipy import stats
F = sys.argv[1]; D = sys.argv[2] if len(sys.argv) > 2 else "results"
model = json.load(open(F))["model"]; items = json.load(open(F))["items"]
key = {int(x["id"]): x for x in csv.DictReader(open(f"{D}/human_eval_v2_key.csv", encoding="utf-8"))}
def rf(a, b): return f"{D}/human_eval_v2_rater{a}.csv" if os.path.exists(f"{D}/human_eval_v2_rater{a}.csv") else f"{D}/human_eval_v2_rater{b}.csv"
RA = {int(x["id"]): int(x["human_faithfulness"]) for x in csv.DictReader(open(rf("A", 2), encoding="utf-8"))}
RB = {int(x["id"]): int(x["human_faithfulness"]) for x in csv.DictReader(open(rf("B", 3), encoding="utf-8"))}
L = {i: int(float(key[i]["judge_faithfulness_v2"])) for i in key}
O = {int(i): int(v["faithfulness"]) for i, v in items.items() if v.get("faithfulness") is not None}
ids = sorted(O); ins = [i for i in ids if key[i]["inserted"] == "1"]; non = [i for i in ids if key[i]["inserted"] != "1"]
print(f"second judge: {model}; scored {len(ids)} of 60 answers ({60-len(ids)} without a score: extraction or verification failure)")
def pr(a, b):
    a, b = np.array(a, float), np.array(b, float); return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
def wk(a, b):
    M = np.zeros((5, 5))
    for x, y in zip(a, b): M[x - 1, y - 1] += 1
    E = np.outer(M.sum(1), M.sum(0)) / M.sum(); W = np.abs(np.subtract.outer(range(5), range(5))) / 4; d = (W * E).sum()
    return float("nan") if d == 0 else 1 - (W * M).sum() / d
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
for nm, R in (("rater A", RA), ("rater B", RB), ("first judge (llama3)", L)):
    a = [O[i] for i in ids]; b = [R[i] for i in ids]
    print(f"{model} vs {nm}: r={pr(a,b):.2f}  weighted kappa={wk(a,b):.2f}  exact={np.mean(np.array(a)==np.array(b)):.2f}")
print("score 5 share: %.0f%%" % (100 * np.mean([O[i] == 5 for i in ids])))
k1 = sum(O[i] < 5 for i in ins); k0 = sum(O[i] < 5 for i in non)
lo, hi = wilson(k1, len(ins)) if ins else (float("nan"),) * 2
p = stats.fisher_exact([[k1, len(ins) - k1], [k0, len(non) - k0]])[1] if ins and non else float("nan")
print(f"planted answers scored below 5: {k1}/{len(ins)} (Wilson {100*lo:.0f}-{100*hi:.0f}%); unmodified scored below 5: {k0}/{len(non)}; Fisher p={p:.3g}")
