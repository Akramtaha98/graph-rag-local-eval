"""Restore the withheld third-party chunk text in a WORKING COPY of the saved results.
Steps: (1) python3 scripts/fetch_corpus.py and the WHO conversion described in NOTICES.md, so that data/docs/ holds the
original texts; (2) python3 scripts/restore_contexts.py; (3) run the judging scripts with RESULTS_DIR=results_restored.
The script rebuilds the chunks with src/ingest.py, matches each placeholder by chunk id AND the SHA-256 prefix stored in the
placeholder, refuses to continue on any mismatch, and writes results_restored/ (originals in results/ are untouched).
Answers, triples, scores and context order are preserved exactly; nothing is regenerated."""
import hashlib, json, os, re, shutil, sys
sys.path.insert(0, ".")
from config import BASE_DIR, RESULTS_DIR
from src.ingest import build_chunks

PH = re.compile(r"id=([0-9a-f]{12}) sha256\[:12\]=([0-9a-f]{12})")
chunks = {c["chunk_id"]: c["text"] for c in build_chunks()}
bad, done = [], 0


def fix(o):
    global done
    if isinstance(o, dict):
        if "chunk_id" in o and "text" in o and "chunk text withheld" in o["text"]:
            m = PH.search(o["text"])
            cid, h = m.group(1), m.group(2)
            t = chunks.get(cid)
            if t is None or hashlib.sha256(t.encode()).hexdigest()[:12] != h:
                bad.append(cid)
            else:
                o["text"] = t; done += 1
        for v in o.values(): fix(v)
    elif isinstance(o, list):
        for v in o: fix(v)


out = os.path.join(BASE_DIR, "results_restored")
shutil.copytree(RESULTS_DIR, out, dirs_exist_ok=True)
for name in ("detailed.json", "detailed_v2.json"):
    d = json.load(open(os.path.join(RESULTS_DIR, name), encoding="utf-8")); fix(d)
    json.dump(d, open(os.path.join(out, name), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
if bad:
    sys.exit(f"{len(set(bad))} chunk(s) do not match their recorded hash (wrong corpus version?): {sorted(set(bad))[:5]} ... results_restored/ is NOT usable.")
print(f"restored {done} chunk texts into results_restored/. Run the judging scripts with RESULTS_DIR=results_restored")
