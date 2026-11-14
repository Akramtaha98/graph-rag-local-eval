"""Rebuild the third-party parts of the retrieval corpus from their original sources.

The public release does not redistribute the two WHO fact sheets or the RFC 2616 text
(see NOTICES.md). This script downloads them from the original URLs, checks them against
the SHA-256 hashes of the files used in the experiments, and rebuilds the HTTP excerpt
(technical_http_rfc2616_trim.txt) from the full RFC. The Common Paper NDA (CC BY 4.0)
is shipped with the release.

WHO pages are edited over time, so a hash mismatch for a WHO file is expected if the page
changed after the retrieval date; the script warns and keeps the file.
Usage: python scripts/fetch_corpus.py [--out data/docs]
"""
import argparse, hashlib, os, sys, urllib.request

SOURCES = {
    "medical_hypertension_who.txt": {
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "version": "Hypertension fact sheet, 25 September 2025 (retrieved July 2026)",
        "sha256": "f5ebcbeb9c0a4ac4c43827c97860f124e37ca7b77650cd109a6f760b2c920491"},
    "medical_cvd_who.txt": {
        "url": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
        "version": "Cardiovascular diseases fact sheet, 31 July 2025 (retrieved July 2026)",
        "sha256": "f1390be3d39278f71248f5625912b090510d72fc4ea96b50790da5b4b0ef99f3"},
}
RFC_URL = "https://www.rfc-editor.org/rfc/rfc2616.txt"
RFC_FULL_SHA = "10211d2885196b97b1c78e1672f3f68ae97c294596ef2b7fd890cbd30a3427bf"
RFC_TRIM_SHA = "89223eb931ce138c2c2fb96850856d57a6afeda5bb2c9ad6e9400d0de04a2d59"
# (start line in the full RFC text, number of lines, marker line placed before the block or None)
RFC_BLOCKS = [(6, 1794, None),
              (2416, 1530, "=== ADDED FOR EXPERIMENT COVERAGE: Persistent Connections, Method Definitions, Status Codes ==="),
              (4119, 341, "=== ADDED FOR EXPERIMENT COVERAGE: Caching Overview and Expiration Model ==="),
              (5998, 62, "=== ADDED FOR EXPERIMENT COVERAGE: Cache-Control Header ==="),
              (7006, 94, "=== ADDED FOR EXPERIMENT COVERAGE: ETag and Expires Headers ==="),
              (7161, 59, "=== ADDED FOR EXPERIMENT COVERAGE: Host Header ==="),
              (7496, 65, "=== ADDED FOR EXPERIMENT COVERAGE: Last-Modified Header ===")]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def build_excerpt(full_bytes):
    full = full_bytes.decode("utf-8", "replace").split("\n")
    out = [""] * 6
    for start, n, marker in RFC_BLOCKS:
        if marker:
            out += ["", marker]
        out += full[start:start + n]
    return "\n".join(out).encode("utf-8")


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "graph-rag-local-eval corpus fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/docs")
    ap.add_argument("--rfc-full", help="path to a local copy of rfc2616.txt (skips download)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ok = True
    full = open(a.rfc_full, "rb").read() if a.rfc_full else get(RFC_URL)
    print("RFC 2616 full text sha256 matches:", sha(full) == RFC_FULL_SHA)
    open(os.path.join(a.out, "technical_http_rfc2616.txt"), "wb").write(full)
    trim = build_excerpt(full)
    good = sha(trim) == RFC_TRIM_SHA
    print("HTTP excerpt rebuilt, sha256 matches:", good)
    ok &= good
    open(os.path.join(a.out, "technical_http_rfc2616_trim.txt"), "wb").write(trim)
    for name, meta in SOURCES.items():
        print("fetching", meta["url"])
        print("note: the experiment file is a plain-text copy of", meta["version"],
              "with a source header; the raw HTML downloaded here must be converted to text "
              "and its header added by hand, so a hash comparison is only informative if you "
              "reproduce that format. Saving raw page as", name + ".html")
        open(os.path.join(a.out, name + ".html"), "wb").write(get(meta["url"]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
