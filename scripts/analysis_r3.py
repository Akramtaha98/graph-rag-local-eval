"""Third-round checks: refusal sensitivity of the judge's flags, identity of the 48 unmodified scores across runs, flag coincidence.
Run from the package root: python3 scripts/analysis_r3.py  -> results/analysis_r3.json"""
import csv, json
from scipy.stats import fisher_exact
R = "results/"
key = {r["id"]: r for r in csv.DictReader(open(R + "human_eval_v2_key.csv", encoding="utf-8"))}
rd = lambda f: {x["id"]: int(x["human_faithfulness"]) for x in csv.DictReader(open(R + f, encoding="utf-8"))}
A, B = rd("human_eval_v2_raterA.csv"), rd("human_eval_v2_raterB.csv")
det = json.load(open(R + "detailed_v2.json"))
un = [i for i in key if key[i]["inserted"] == "0"]
J = {i: float(key[i]["judge_faithfulness_v2"]) for i in key if key[i]["judge_faithfulness_v2"] not in ("", "None")}
fl = [i for i in un if i in J and J[i] < 5]
refusals = ["5", "22"]
fl_x = [i for i in fl if i not in refusals]
n_x = len(un) - len(refusals)
out = {"judge_flags_unmodified": fl, "n_unmodified": len(un),
       "excluding_refusals": {"flags": len(fl_x), "n": n_x, "fisher_p_vs_2_of_12": fisher_exact([[2, 10], [len(fl_x), n_x - len(fl_x)]])[1]},
       "rater_A_flags": [i for i in un if A[i] < 5], "rater_B_flags": [i for i in un if B[i] < 5]}
out["overlap_judge_A"] = sorted(set(fl) & set(out["rater_A_flags"])); out["overlap_judge_B"] = sorted(set(fl) & set(out["rater_B_flags"]))
out["unmodified_scores_identical_to_main_run"] = sum(det[int(key[i]["orig_index"])]["v2_faithfulness"] == J.get(i) for i in un)
json.dump(out, open(R + "analysis_r3.json", "w"), indent=1); print(json.dumps(out, indent=1))
