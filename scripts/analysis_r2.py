"""Second-round analyses of the fresh validation sample (intervals, discrimination, thresholds, run identity).
Usage: python3 scripts/analysis_r2.py [results_dir]   (needs numpy and scipy; no model is called)"""
import csv, json, sys, math, os
import numpy as np
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else "results"
key = {int(x["id"]): x for x in csv.DictReader(open(f"{D}/human_eval_v2_key.csv", encoding="utf-8"))}
def rf(a, b): return f"{D}/human_eval_v2_rater{a}.csv" if os.path.exists(f"{D}/human_eval_v2_rater{a}.csv") else f"{D}/human_eval_v2_rater{b}.csv"
RA = {int(x["id"]): int(x["human_faithfulness"]) for x in csv.DictReader(open(rf("A", 2), encoding="utf-8"))}
RB = {int(x["id"]): int(x["human_faithfulness"]) for x in csv.DictReader(open(rf("B", 3), encoding="utf-8"))}
det = json.load(open(f"{D}/judge_sample_v2_details.json"))
ids = sorted(key)
J = {i: int(float(key[i]["judge_faithfulness_v2"])) for i in ids}
ins = [i for i in ids if key[i]["inserted"] == "1"]; non = [i for i in ids if key[i]["inserted"] != "1"]


def score(f, t=0.9): return 5 if f >= t else 4 if f >= .7 else 3 if f >= .5 else 2 if f >= .25 else 1
def wk(a, b):
    a, b = np.array(a, int), np.array(b, int); O = np.zeros((5, 5))
    for x, y in zip(a, b): O[x - 1, y - 1] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum(); W = np.abs(np.subtract.outer(range(5), range(5))) / 4
    d = (W * E).sum(); return float("nan") if d == 0 else 1 - (W * O).sum() / d
def pr(a, b):
    a, b = np.array(a, float), np.array(b, float); return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
def sp(a, b): return float(stats.spearmanr(a, b)[0])
def ac1(a, b):  # Gwet AC1, nominal, with the fixed category set 1-5
    a, b = np.array(a), np.array(b); n = len(a); cats = [1, 2, 3, 4, 5]; Pa = np.mean(a == b)
    pi = [(np.mean(a == c) + np.mean(b == c)) / 2 for c in cats]; Pe = sum(p * (1 - p) for p in pi) / (len(cats) - 1)
    return (Pa - Pe) / (1 - Pe)
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return c - h, c + h
def boot(fn, a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed); a, b = np.array(a), np.array(b); v = []
    for _ in range(n):
        ix = rng.integers(0, len(a), len(a)); x = fn(a[ix], b[ix])
        if not math.isnan(x): v.append(x)
    return np.percentile(v, [2.5, 97.5])
def mcnemar(b, c): return float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0

out = {}
print("== Agreement with 95% bootstrap intervals (10,000 resamples, seed 0) ==")
pairs = {"judge_vs_A": (J, RA), "judge_vs_B": (J, RB), "A_vs_B": (RA, RB)}
for name, (x, y) in pairs.items():
    a = [x[i] for i in ids]; b = [y[i] for i in ids]
    r = pr(a, b); lo, hi = boot(pr, a, b); k = wk(a, b); klo, khi = boot(wk, a, b)
    rho = sp(a, b); rlo, rhi = boot(sp, a, b)
    out[name] = dict(r=r, r_ci=[lo, hi], kappa=k, kappa_ci=[klo, khi], rho=rho, rho_ci=[rlo, rhi], ac1=ac1(a, b), exact=float(np.mean(np.array(a) == np.array(b))))
    print(f"{name}: r={r:.2f} [{lo:.2f},{hi:.2f}]  kappa_w={k:.2f} [{klo:.2f},{khi:.2f}]  rho={rho:.2f} [{rlo:.2f},{rhi:.2f}]  AC1={ac1(a,b):.2f}  exact={out[name]['exact']:.2f}")
print("prevalence of score 5: judge %.0f%%  A %.0f%%  B %.0f%%" % tuple(100 * np.mean([s[i] == 5 for i in ids]) for s in (J, RA, RB)))

print("\n== Detection of inserted sentences and false-alarm rate (score below 5) ==")
sc = {"judge": J, "A": RA, "B": RB}
for n, s in sc.items():
    k1 = sum(s[i] < 5 for i in ins); k0 = sum(s[i] < 5 for i in non)
    lo, hi = wilson(k1, len(ins)); lo0, hi0 = wilson(k0, len(non))
    p = stats.fisher_exact([[k1, len(ins) - k1], [k0, len(non) - k0]])[1]
    out["detect_" + n] = dict(planted=k1, planted_n=len(ins), planted_ci=[lo, hi], unmod=k0, unmod_n=len(non), unmod_ci=[lo0, hi0], fisher_p=float(p))
    print(f"{n}: planted {k1}/{len(ins)} ({100*k1/len(ins):.0f}%, Wilson {100*lo:.0f}-{100*hi:.0f}%); unmodified {k0}/{len(non)} ({100*k0/len(non):.0f}%, Wilson {100*lo0:.0f}-{100*hi0:.0f}%); Fisher p={p:.3g}")
print("\n== Exact McNemar on the 12 planted items (judge vs rater) ==")
for n in ("A", "B"):
    s = sc[n]; b = sum(s[i] < 5 and J[i] == 5 for i in ins); c = sum(J[i] < 5 and s[i] == 5 for i in ins)
    p2 = mcnemar(b, c); p1 = p2 / 2 if b != c else 1.0
    out["mcnemar_" + n] = dict(rater_only=b, judge_only=c, p_two=p2, p_one=p1)
    print(f"judge vs {n}: rater-only flags {b}, judge-only flags {c}, two-sided p={p2:.3f}, one-sided p={p1:.3f}")

print("\n== Run identity: reported judge scores vs the stored-claims re-run ==")
same = sum(J[i] == score(det[str(i)]["found_fraction"]) for i in ids if det[str(i)].get("found_fraction") is not None)
print(f"identical item scores in both runs: {same} of {len(ids)}"); out["run_identity"] = [same, len(ids)]

print("\n== Stricter threshold (score 5 requires found fraction at or above t) ==")
for t in (0.9, 0.95, 1.0):
    s = {i: score(det[str(i)]["found_fraction"], t) for i in ids}
    k1 = sum(s[i] < 5 for i in ins); k0 = sum(s[i] < 5 for i in non); lo, hi = wilson(k1, len(ins))
    pr_ = stats.fisher_exact([[k1, len(ins) - k1], [k0, len(non) - k0]])[1]
    out[f"thr_{t}"] = dict(planted=k1, unmod=k0, fisher_p=float(pr_))
    print(f"t={t}: planted flagged {k1}/12 (Wilson {100*lo:.0f}-{100*hi:.0f}%), unmodified flagged {k0}/48, Fisher p={pr_:.3g}")
json.dump(out, open(f"{D}/analysis_r2.json", "w"), indent=1)
