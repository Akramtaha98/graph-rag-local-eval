"""Re-analysis of released data to address reviewer comments (run with python3 -I analysis.py <project_dir>)."""
import csv, json, sys, re, pickle, collections
import numpy as np
from scipy import stats

P = sys.argv[1]
R = f"{P}/results"
det = json.load(open(f"{R}/detailed.json"))
key = {int(r["id"]): r for r in csv.DictReader(open(f"{R}/human_eval_key.csv"))}
r1 = {int(r["id"]): r for r in csv.DictReader(open(f"{R}/human_eval_blind.csv"))}
r2 = {int(r["id"]): r for r in csv.DictReader(open(f"{R}/human_eval_blind_rater2.csv"))}


def wkappa(a, b, cats=(1, 2, 3, 4, 5)):
    a = np.array(a, int); b = np.array(b, int)
    k = len(cats); idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[idx[x], idx[y]] += 1
    n = O.sum()
    E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    den = (W * E).sum()
    return float("nan") if den == 0 else 1 - (W * O).sum() / den


def pear(a, b):
    a = np.array(a, float); b = np.array(b, float)
    if a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def agree(h, j):
    h = np.array(h, float); j = np.array(j, float)
    return dict(n=len(h), r=pear(h, j), exact=float((h == j).mean()), w1=float((abs(h - j) <= 1).mean()),
                mae=float(abs(h - j).mean()), bias=float((j - h).mean()))


out = {}
rows = []
for i, k in key.items():
    lf = k["llm_faithfulness"].strip()
    rows.append(dict(id=i, system=k["system"], type=k["type"], j=float(lf) if lf else None,
                     h1=int(r1[i]["human_faithfulness"]), h2=int(r2[i]["human_faithfulness"])))
for r in rows: r["avg"] = (r["h1"] + r["h2"]) / 2
valid = [r for r in rows if r["j"] is not None]
retr = [r for r in valid if r["system"] != "no_rag"]
out["n_sample"] = len(rows); out["n_valid"] = len(valid); out["n_retr_valid"] = len(retr)
out["n_norag_valid"] = len(valid) - len(retr)

# Comment 1: tables
tab = {}
for name, sub in [("all", valid), ("retr", retr)]:
    tab[f"judge_vs_R1_{name}"] = agree([r["h1"] for r in sub], [r["j"] for r in sub])
    tab[f"judge_vs_avg_{name}"] = agree([r["avg"] for r in sub], [r["j"] for r in sub])
    tab[f"judge_vs_R1_{name}"]["wkappa"] = wkappa([r["h1"] for r in sub], [round(r["j"]) for r in sub])
    tab[f"judge_vs_R2_{name}"] = agree([r["h2"] for r in sub], [r["j"] for r in sub])
    tab[f"judge_vs_R2_{name}"]["wkappa"] = wkappa([r["h2"] for r in sub], [round(r["j"]) for r in sub])
# constant baselines on retrieving rows (R1)
for c in (5, 4):
    tab[f"const{c}_vs_R1_retr"] = agree([r["h1"] for r in retr], [c] * len(retr))
    tab[f"const{c}_vs_avg_retr"] = agree([r["avg"] for r in retr], [c] * len(retr))
    tab[f"const{c}_vs_R1_retr_sameValid"] = tab[f"const{c}_vs_R1_retr"]
# inter-rater
allr = rows
retr_all = [r for r in rows if r["system"] != "no_rag"]
tab["R1_vs_R2_all"] = dict(n=len(allr), exact=float(np.mean([r["h1"] == r["h2"] for r in allr])),
                           w1=float(np.mean([abs(r["h1"] - r["h2"]) <= 1 for r in allr])),
                           wkappa=wkappa([r["h1"] for r in allr], [r["h2"] for r in allr]),
                           r=pear([r["h1"] for r in allr], [r["h2"] for r in allr]))
tab["R1_vs_R2_retr"] = dict(n=len(retr_all), exact=float(np.mean([r["h1"] == r["h2"] for r in retr_all])),
                            w1=float(np.mean([abs(r["h1"] - r["h2"]) <= 1 for r in retr_all])),
                            wkappa=wkappa([r["h1"] for r in retr_all], [r["h2"] for r in retr_all]),
                            r=pear([r["h1"] for r in retr_all], [r["h2"] for r in retr_all]))
out["agreement"] = tab
out["R1_dist_retr"] = dict(collections.Counter(r["h1"] for r in retr_all))
out["R2_dist_retr"] = dict(collections.Counter(r["h2"] for r in retr_all))
out["R1_dist_retr_valid"] = dict(collections.Counter(r["h1"] for r in retr))
out["judge_dist_retr"] = dict(collections.Counter(r["j"] for r in retr))
# confusion matrix judge (rows) x R1 (cols) on retrieving rows
cm = np.zeros((5, 5), int)
for r in retr: cm[int(r["j"]) - 1, r["h1"] - 1] += 1
out["confusion_judge_rows_R1_cols_retr"] = cm.tolist()
cm2 = np.zeros((5, 5), int)
for r in retr_all: cm2[r["h2"] - 1, r["h1"] - 1] += 1
out["confusion_R2_rows_R1_cols_retr"] = cm2.tolist()
# Spearman on retrieving
out["spearman_retr_R1"] = float(stats.spearmanr([r["h1"] for r in retr], [r["j"] for r in retr])[0])
# binary: human<5 detection by judge<5
tp = sum(1 for r in retr if r["h1"] < 5 and r["j"] < 5); fn = sum(1 for r in retr if r["h1"] < 5 and r["j"] == 5)
fp = sum(1 for r in retr if r["h1"] == 5 and r["j"] < 5); tn = sum(1 for r in retr if r["h1"] == 5 and r["j"] == 5)
out["binary_h<5_vs_j<5"] = dict(tp=tp, fn=fn, fp=fp, tn=tn)
# Decomposition: contribution of no_rag
# (a) agreement with judge restricted to retrieving rows already above; (b) all rows with no_rag judge-score replaced by constant 1 (code-set)
out["norag_rows_in_valid"] = [(r["j"], r["h1"], r["h2"]) for r in valid if r["system"] == "no_rag"][:20]

# Noise floor (4.4.1)
res = np.array([r["j"] - r["avg"] for r in retr])
sd = res.std(ddof=1)
out["noise"] = dict(n=len(res), mean=float(res.mean()), sd=float(sd))
for n in (35, 36, 41):
    out["noise"][f"se_n{n}"] = float(sd * np.sqrt(2 / n))
bys = {}
for s in ("naive_rag", "bm25_hybrid", "graph_rag"):
    rr = np.array([r["j"] - r["avg"] for r in retr if r["system"] == s])
    bys[s] = dict(n=len(rr), mean=float(rr.mean()), sd=float(rr.std(ddof=1)), se=float(rr.std(ddof=1) / np.sqrt(len(rr))))
    # vs R1 only
    rr1 = np.array([r["j"] - r["h1"] for r in retr if r["system"] == s])
    bys[s]["mean_vsR1"] = float(rr1.mean())
out["bias_by_system"] = bys

# Pairwise tests (4.1), graph vs naive etc.
by = collections.defaultdict(dict)
for k, row in enumerate(det):
    by[row["question"]][row["system"]] = row
qs = list(by)
rng = np.random.default_rng(0)


def paired(a, b, qsub, field="faithfulness"):
    x = []; y = []
    for q in qsub:
        va = by[q][a][field]; vb = by[q][b][field]
        if va is None or vb is None: continue
        x.append(va); y.append(vb)
    x = np.array(x, float); y = np.array(y, float); d = x - y
    if len(d) == 0: return None
    bs = rng.choice(d, (10000, len(d))).mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # bootstrap p (two-sided, centred)
    cent = d - d.mean()
    bsn = rng.choice(cent, (10000, len(d))).mean(1)
    p = float((abs(bsn) >= abs(d.mean())).mean())
    try:
        w = stats.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    except Exception:
        w = float("nan")
    return dict(n=len(d), diff=float(d.mean()), lo=float(lo), hi=float(hi), p_boot=p, p_wilcoxon=float(w),
                n_zero=int((d == 0).sum()))


typ = {q: by[q]["naive_rag"]["type"] for q in qs}
mh = [q for q in qs if typ[q] == "multi_hop"]
pairs = [("naive_rag", "bm25_hybrid"), ("naive_rag", "graph_rag"), ("bm25_hybrid", "graph_rag")]
out["pairwise_all"] = {f"{a}-{b}": paired(a, b, qs) for a, b in pairs}
out["pairwise_multihop"] = {f"{a}-{b}": paired(a, b, mh) for a, b in pairs}

# Per-system means / failures
summ = {}
for s in ("no_rag", "naive_rag", "bm25_hybrid", "graph_rag"):
    v = [by[q][s]["faithfulness"] for q in qs]
    ok = [x for x in v if x is not None]
    summ[s] = dict(mean=float(np.mean(ok)), fails=len(v) - len(ok), n_ok=len(ok),
                   rel=float(np.mean([by[q][s]["relevance"] for q in qs if by[q][s]["relevance"] is not None])),
                   cp=float(np.mean([by[q][s]["context_precision"] for q in qs if by[q][s]["context_precision"] is not None])))
out["summary"] = summ

# Refusals / zero-claim rows
ref = [(r["system"], r["question"][:60], r["faithfulness"]) for r in det if "no factual claims" in (r.get("justification") or "")]
out["zero_claim_rows"] = ref
refus = [r for r in det if re.search(r"(does not|doesn't|do not|not) (contain|provide|mention|specify|include|state|say)|cannot (answer|be determined)|no information|not (available|mentioned|specified|provided) in the (context|provided)|context (does not|doesn't|lacks)", r["answer"], re.I) and r["system"] != "no_rag"]
out["refusal_like_by_system"] = dict(collections.Counter(r["system"] for r in refus))
out["refusal_like_scores"] = [(r["system"], r["faithfulness"]) for r in refus]

# Raw found-fraction sensitivity
frac = {}
for r in det:
    m = re.search(r"(\d+)/(\d+) claims grounded", r.get("justification") or "")
    if m and r["system"] != "no_rag":
        frac[(r["question"], r["system"])] = int(m.group(1)) / int(m.group(2))
out["n_frac"] = len(frac)
raw = {}
for s in ("naive_rag", "bm25_hybrid", "graph_rag"):
    v = [frac[(q, s)] for q in qs if (q, s) in frac]
    raw[s] = dict(n=len(v), mean_frac=float(np.mean(v)), mean_frac_x4p0_plus1=float(1 + 4 * np.mean(v)))
out["raw_fraction"] = raw


def paired_frac(a, b, qsub):
    d = [frac[(q, a)] - frac[(q, b)] for q in qsub if (q, a) in frac and (q, b) in frac]
    d = np.array(d)
    bs = rng.choice(d - d.mean(), (10000, len(d))).mean(1)
    return dict(n=len(d), diff=float(d.mean()), p_boot=float((abs(bs) >= abs(d.mean())).mean()))


out["raw_fraction_pairwise"] = {f"{a}-{b}": paired_frac(a, b, qs) for a, b in pairs}
# mean number of claims
nclaims = collections.defaultdict(list)
for r in det:
    m = re.search(r"(\d+)/(\d+) claims grounded", r.get("justification") or "")
    if m: nclaims[r["system"]].append(int(m.group(2)))
out["claims_per_answer"] = {s: float(np.mean(v)) for s, v in nclaims.items()}

# Cross-document subset: find multi-hop questions mentioning both public-health sheets
def is_cross(q):
    t = q.lower()
    hyp = any(w in t for w in ("hypertension", "blood pressure"))
    cvd = any(w in t for w in ("cardiovascular", "cvd", "heart", "stroke", "ischaemic", "ischemic", "rheumatic"))
    return hyp and cvd and typ[q] == "multi_hop"


cross_cand = [q for q in mh if is_cross(q)]
out["cross_candidates"] = cross_cand
cd = {}
for q in cross_cand:
    cd[q] = {s: by[q][s]["faithfulness"] for s in ("no_rag", "naive_rag", "bm25_hybrid", "graph_rag")}
out["cross_scores"] = cd

# Context sizes
cs = collections.defaultdict(list)
for r in det:
    c = r.get("context", {})
    txt = "\n".join(f"[{x['source']}] {x['text']}" for x in c.get("chunks", [])) if c else ""
    tri = "\n".join(f"{t['subject']} --{t['relation']}--> {t['object']}" for t in c.get("triples", []))
    cs[r["system"]].append((len(txt), len(tri), len(txt + "\n" + tri)))
out["context_chars"] = {s: [float(np.mean([x[i] for x in v])) for i in range(3)] for s, v in cs.items()}
out["judge_truncation_6000_share"] = {s: float(np.mean([x[2] > 6000 for x in v])) for s, v in cs.items()}

# Corpus stats
import os
docs = {}
for f in sorted(os.listdir(f"{P}/data/docs")):
    if f == "technical_http_rfc2616.txt": continue
    docs[f] = len(open(f"{P}/data/docs/{f}", encoding="utf-8", errors="ignore").read().split())
out["doc_words"] = docs
ch = pickle.load(open(f"{P}/data/index/chunks.pkl", "rb"))
out["chunks_per_doc"] = dict(collections.Counter(c["source"] for c in ch))
wl = [len(c["text"].split()) for c in ch]
out["chunk_words"] = dict(mean=float(np.mean(wl)), max=int(max(wl)))
# crude token estimate: chars/ 4 and words*1.3 (no tokenizer offline)
out["chunk_est_tokens_over256_words*1.3"] = float(np.mean([w * 1.3 > 256 for w in wl]))
out["chunk_est_tokens_over256_chars/4"] = float(np.mean([len(c["text"]) / 4 > 256 for c in ch]))
g = pickle.load(open(f"{P}/data/index/knowledge_graph.pkl", "rb"))
out["kg"] = dict(nodes=g.number_of_nodes(), edges=g.number_of_edges(), triples_per_chunk=g.number_of_edges() / len(ch))

# Identical-chunk check naive vs graph
same = 0
for q in qs:
    a = [c["chunk_id"] if "chunk_id" in c else c["text"] for c in by[q]["naive_rag"]["context"]["chunks"]]
    b = [c["chunk_id"] if "chunk_id" in c else c["text"] for c in by[q]["graph_rag"]["context"]["chunks"]]
    same += (a == b)
out["identical_chunks_naive_graph"] = same

# Failure type / rate chi2
f = np.array([[summ[s]["fails"], summ[s]["n_ok"]] for s in ("naive_rag", "bm25_hybrid", "graph_rag")])
out["fail_chi2_p"] = float(stats.chi2_contingency(f)[1])

json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
