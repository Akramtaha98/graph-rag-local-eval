"""Where does the revised judge lose the 12 inserted unsupported sentences?
Matches each inserted sentence to the stored claims (share of its words of 4+ letters found in a claim; >= 0.5 = extracted),
then reports the verdict on that claim and the final score. Run after judge_sample_v2.py. Usage: python3 scripts/planted_audit.py"""
import json, csv, re
R = "results/"
d = json.load(open(R + "judge_sample_v2_details.json"))
key = {x["id"]: x for x in csv.DictReader(open(R + "human_eval_v2_key.csv", encoding="utf-8"))}
tok = lambda s: set(re.findall(r"[a-z0-9]{4,}", s.lower()))
score = lambda f: 5 if f >= .9 else 4 if f >= .7 else 3 if f >= .5 else 2 if f >= .25 else 1
rows = []
for i, x in key.items():
    if x["inserted"] not in ("1", "True", "true", "yes"): continue
    r = d[i]; it = tok(x["inserted_text"]); best, bi = 0, None
    for j, c in enumerate(r["claims"]):
        o = len(tok(c) & it) / max(1, len(it))
        if o > best: best, bi = o, j
    ext = best >= .5
    v = r["verdicts"][bi] if ext else ""
    s = score(r["found_fraction"])
    stage = "not extracted" if not ext else ("verified as found" if v == "FOUND" else ("flagged, score still 5" if s == 5 else "flagged, score below 5"))
    rows.append(dict(id=i, n_claims=len(r["claims"]), best_overlap=round(best, 2), verdict=v, found_fraction=round(r["found_fraction"], 3), score=s, outcome=stage))
w = csv.DictWriter(open(R + "planted_audit.csv", "w", newline=""), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r["outcome"] for r in rows))
