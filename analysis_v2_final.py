"""Final numbers from the re-judged run (detailed_v2.json) and the fresh human sample.
Usage: python3 analysis_v2_final.py <project_dir> <out.json>"""
import sys, json, csv, collections, itertools
import numpy as np
from scipy import stats
P, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(f"{P}/results/detailed_v2.json"))
qs = json.load(open(f"{P}/questions.json"))
S = ["naive_rag", "bm25_hybrid", "graph_rag"]
by = {s: {} for s in S}
for r in d:
    if r["system"] in by: by[r["system"]][r["question"]] = r
Q = [x["question"] for x in qs]
sub = {"all": Q, "single": [x["question"] for x in qs if x["type"] == "single_hop"],
       "multi": [x["question"] for x in qs if x["type"] == "multi_hop"], "cross": [qs[i]["question"] for i in range(21, 26)]}
rng = np.random.default_rng(0)
res = {"subsets": {}, "pairs": {}}
def boot(x, B=10000):
    x = np.array(x, float); return [float(v) for v in np.percentile(rng.choice(x, (B, len(x))).mean(1), [2.5, 97.5])]
for k, qq in sub.items():
    for s in S:
        v = [by[s][q]["v2_faithfulness"] for q in qq if by[s][q]["v2_faithfulness"] is not None]
        rel = [by[s][q]["v2_relevance"] for q in qq if by[s][q]["v2_relevance"] is not None]
        cp = [by[s][q]["v2_context_precision"] for q in qq if by[s][q]["v2_context_precision"] is not None]
        ff = [by[s][q]["v2_found_fraction"] for q in qq if by[s][q]["v2_found_fraction"] is not None]
        res["subsets"][f"{k}|{s}"] = dict(n=len(v), of=len(qq), mean=float(np.mean(v)), ci=boot(v), rel=float(np.mean(rel)), cp=float(np.mean(cp)) if cp else None,
                                         ff=float(np.mean(ff)), n5=int(sum(x == 5 for x in v)), scores=[float(x) for x in v] if k == "cross" else None)
def pair(a, b, qq, key="v2_faithfulness"):
    qq = [q for q in qq if by[a][q][key] is not None and by[b][q][key] is not None]
    x = np.array([by[a][q][key] - by[b][q][key] for q in qq], float); n = len(x); m = x.mean()
    ci = [float(v) for v in np.percentile(rng.choice(x, (10000, n)).mean(1), [2.5, 97.5])]
    c = x - m; p = float(np.mean(np.abs(rng.choice(c, (10000, n)).mean(1)) >= abs(m)))
    nz = x[x != 0]; w = float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0
    return dict(n=n, diff=float(m), ci=ci, p=p, wil=w, zeros=int((x == 0).sum()))
for k in ("all", "multi", "single"):
    for a, b in itertools.combinations(S, 2):
        res["pairs"][f"{k}|{a}|{b}"] = pair(a, b, sub[k]); res["pairs"][f"{k}|{a}|{b}|ff"] = pair(a, b, sub[k], "v2_found_fraction")
# coverage / failures / refusals / tokens / context
res["system"] = {}
for s in S:
    rr = list(by[s].values())
    res["system"][s] = dict(ok=sum(r["v2_status"] == "ok" for r in rr), fail=sum(r["v2_status"] != "ok" for r in rr),
        ctx_mean=float(np.mean([r["v2_context_chars"] for r in rr])), max_tok=max(r["v2_max_prompt_tokens"] for r in rr),
        mean_claims=float(np.mean([len(r["v2_claims"]) for r in rr if r["v2_status"] == "ok"])),
        mean_triples=float(np.mean([r["num_triples"] for r in rr])))
# Holm over the 3 overall pairs (found fraction)
ps = sorted([(res["pairs"][f"all|{a}|{b}"]["p"], f"{a}|{b}") for a, b in itertools.combinations(S, 2)])
res["holm"] = {n: min(1.0, p * (3 - i)) for i, (p, n) in enumerate(ps)}
# ---- validation sample ----
key = {int(r["id"]): r for r in csv.DictReader(open(f"{P}/results/human_eval_v2_key.csv", encoding="utf-8"))}
H = {}
for n in (2, 3):
    H[n] = {int(r["id"]): int(r["human_faithfulness"]) for r in csv.DictReader(open(f"{P}/results/human_eval_v2_rater{ {2: 'A', 3: 'B'}[n] }.csv", encoding="utf-8"))}
ids = sorted(key); J = np.array([float(key[i]["judge_faithfulness_v2"]) for i in ids]); ins = np.array([key[i]["inserted"] == "1" for i in ids])
sysn = np.array([key[i]["system"] for i in ids])
r2 = np.array([H[2][i] for i in ids]); r3 = np.array([H[3][i] for i in ids]); mean = (r2 + r3) / 2
def wk(a, b):
    O = np.zeros((5, 5))
    for x, y in zip(a.astype(int), b.astype(int)): O[x - 1, y - 1] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum(); W = np.abs(np.subtract.outer(range(5), range(5))) / 4
    dd = (W * E).sum(); return float("nan") if dd == 0 else float(1 - (W * O).sum() / dd)
def rr(a, b): return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
res["val"] = dict(n=len(ids), n_ins=int(ins.sum()),
    judge_dist={int(k): int(v) for k, v in collections.Counter(J).items()}, r2_dist={int(k): int(v) for k, v in collections.Counter(r2).items()}, r3_dist={int(k): int(v) for k, v in collections.Counter(r3).items()},
    judge_r2=dict(r=rr(J, r2), exact=float((J == r2).mean()), w1=float((abs(J - r2) <= 1).mean()), k=wk(J, r2)),
    judge_r3=dict(r=rr(J, r3), exact=float((J == r3).mean()), w1=float((abs(J - r3) <= 1).mean()), k=wk(J, r3)),
    judge_mean=dict(r=rr(J, mean), w1=float((abs(J - mean) <= 1).mean())),
    const_r2=dict(exact=float((r2 == 5).mean()), w1=float((abs(5 - r2) <= 1).mean())), const_r3=dict(exact=float((r3 == 5).mean()), w1=float((abs(5 - r3) <= 1).mean())),
    r2_r3=dict(r=rr(r2, r3), exact=float((r2 == r3).mean()), w1=float((abs(r2 - r3) <= 1).mean()), k=wk(r2, r3)),
    ins_flag=dict(judge=int((J[ins] < 5).sum()), r2=int((r2[ins] < 5).sum()), r3=int((r3[ins] < 5).sum()), either=int(((r2[ins] < 5) | (r3[ins] < 5)).sum()), both=int(((r2[ins] < 5) & (r3[ins] < 5)).sum())),
    clean_flag=dict(judge=int((J[~ins] < 5).sum()), r2=int((r2[~ins] < 5).sum()), r3=int((r3[~ins] < 5).sum()), n=int((~ins).sum())),
    means=dict(judge=float(J.mean()), r2=float(r2.mean()), r3=float(r3.mean())),
    per_system={s: dict(n=int((sysn == s).sum()), judge=float(J[sysn == s].mean()), rater_mean=float(mean[sysn == s].mean())) for s in S},
    judge_status=dict(collections.Counter(key[i]["judge_status"] for i in ids)))
# rating the judge on the judged-5 rows: how many had raters <5
res["val"]["judge5_rater_lt5"] = dict(r2=int(((J == 5) & (r2 < 5)).sum()), r3=int(((J == 5) & (r3 < 5)).sum()), n=int((J == 5).sum()))
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(json.dumps(res["val"], indent=1, default=float)); print(json.dumps(res["system"], indent=1, default=float)); print(res["holm"])
for k, v in res["subsets"].items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "scores"})
for k, v in res["pairs"].items(): print(k, v)
