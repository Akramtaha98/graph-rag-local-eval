"""Agreement analysis for the fresh round (what Table 5/6 of the revised paper should be re-filled with).
Reports, on retrieving systems only: judge vs each rater, judge vs rater mean, constant-score baseline,
rater-rater weighted kappa, confusion matrices, and detection of inserted unsupported sentences.
Usage: python scripts/analyze_v2.py [results_dir]   (needs human_eval_v2_key.csv with judge scores + rater CSVs filled)"""
import csv, sys, glob, itertools
import numpy as np
from collections import Counter
D = sys.argv[1] if len(sys.argv) > 1 else "results"


def wk(a, b):
    a, b = np.array(a, int), np.array(b, int)
    O = np.zeros((5, 5))
    for x, y in zip(a, b): O[x - 1, y - 1] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    W = np.abs(np.subtract.outer(range(5), range(5))) / 4
    d = (W * E).sum()
    return float("nan") if d == 0 else 1 - (W * O).sum() / d


def r(a, b):
    a, b = np.array(a, float), np.array(b, float)
    return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def row(name, h, j):
    h, j = np.array(h, float), np.array(j, float)
    return f"| {name} | {len(h)} | {r(h, j):.2f} | {100*np.mean(h==j):.0f} | {100*np.mean(abs(h-j)<=1):.0f} | {wk(h, j):.2f} |"


key = {int(x["id"]): x for x in csv.DictReader(open(f"{D}/human_eval_v2_key.csv", encoding="utf-8"))}
raters = []
for p in sorted(glob.glob(f"{D}/human_eval_v2_rater*.csv")):
    d = {int(x["id"]): x for x in csv.DictReader(open(p, encoding="utf-8"))}
    if all(v["human_faithfulness"].strip() for v in d.values()):
        raters.append((p.split("rater")[-1][:-4], {i: int(v["human_faithfulness"]) for i, v in d.items()}))
ids = [i for i in key if key[i]["judge_faithfulness_v2"].strip()]
print(f"items with judge score: {len(ids)}; completed raters: {[n for n, _ in raters]}\n")
print("| Comparison | n | Pearson r | Exact % | Within 1 % | Weighted kappa |\n|---|---|---|---|---|---|")
J = [float(key[i]["judge_faithfulness_v2"]) for i in ids]
for n, h in raters:
    print(row(f"Judge vs rater {n}", [h[i] for i in ids], J))
if raters:
    mean = [np.mean([h[i] for _, h in raters]) for i in ids]
    print(f"| Judge vs rater mean | {len(ids)} | {r(mean, J):.2f} | n/a | {100*np.mean(abs(np.array(mean)-np.array(J))<=1):.0f} | n/a |")
    for n, h in raters:
        maj = Counter(h[i] for i in ids).most_common(1)[0][0]
        print(row(f"Constant {maj} vs rater {n}", [h[i] for i in ids], [maj] * len(ids)))
for (n1, h1), (n2, h2) in itertools.combinations(raters, 2):
    print(row(f"Rater {n1} vs rater {n2}", [h1[i] for i in key], [h2[i] for i in key]))
if raters:
    ins = [i for i in ids if key[i]["inserted"] == "1"]
    non = [i for i in ids if key[i]["inserted"] != "1"]
    for n, h in raters:
        print(f"\nRater {n}: scored inserted rows <5 in {sum(h[i]<5 for i in ins)}/{len(ins)}; non-inserted rows <5 in {sum(h[i]<5 for i in non)}/{len(non)}")
    print(f"Judge: scored inserted rows <5 in {sum(float(key[i]['judge_faithfulness_v2'])<5 for i in ins)}/{len(ins)}; "
          f"non-inserted rows <5 in {sum(float(key[i]['judge_faithfulness_v2'])<5 for i in non)}/{len(non)}")
    print("\nConfusion (rows = judge 1..5, cols = rater mean rounded 1..5):")
    M = np.zeros((5, 5), int)
    for i, m in zip(ids, mean): M[int(float(key[i]["judge_faithfulness_v2"])) - 1, int(round(m)) - 1] += 1
    print(M)
