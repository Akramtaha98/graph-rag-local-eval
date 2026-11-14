# Third-party materials and licences

This release contains original code, questions, model outputs and human ratings (MIT licence, see LICENSE). The retrieval corpus is built from third-party texts. Their terms differ, so the release treats them differently. This notice is information, not legal advice.

| Text | Source | Terms (as we read them) | Treatment in the release |
|---|---|---|---|
| Common Paper Mutual NDA, Standard Terms v1.0 | https://www.commonpaper.com/standards/mutual-nda/ | Creative Commons CC BY 4.0; attribution to Common Paper and a link to the terms are required | Included in `data/docs/` unchanged, with the attribution header and the licence link https://creativecommons.org/licenses/by/4.0/ |
| WHO "Hypertension" fact sheet, 25 Sep 2025 | https://www.who.int/news-room/fact-sheets/detail/hypertension | WHO website terms allow extracts for research or private study, not for sale or commercial purposes, with acknowledgment of WHO and the URL; reproduction of substantial portions or other uses needs written authorization. Whether the CC BY-NC-SA 3.0 IGO licence that WHO applies to its publications also covers web fact sheets is unclear to us | Not redistributed. URL, version date, and SHA-256 of the file we used are given; the file must be fetched by the user |
| WHO "Cardiovascular diseases (CVDs)" fact sheet, 31 Jul 2025 | https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds) | As above | As above |
| RFC 2616, "Hypertext Transfer Protocol -- HTTP/1.1" (Fielding et al., 1999) | https://www.rfc-editor.org/rfc/rfc2616.txt | Copyright (C) The Internet Society (1999). The document's copyright statement permits copying and derivative works if the notice is kept, but the document itself may not be modified | Not redistributed. `scripts/fetch_corpus.py` downloads the original and rebuilds the HTTP excerpt (the WHO pages are only saved as HTML by the script and must be converted to text by hand; the current pages may differ from the versions we used) (seven verbatim line ranges, plus seven marker lines that we added), checked by SHA-256 |

The reviewer package does not contain the WHO fact sheets or the RFC 2616 text as files, and the chunk index and the logged contexts (results/*.json, rater files) no longer contain their text: each chunk is replaced by its identifier and a SHA-256 prefix. Run scripts/fetch_corpus.py to rebuild the HTTP excerpt exactly (hash-verified); the WHO texts must be converted from the web pages by the user and compared with data/docs_sha256.txt; then run scripts/restore_contexts.py to restore the logged contexts in a working copy before re-running the judge. The Common Paper NDA is included under CC BY 4.0. Model outputs (answers, extracted claims), the 12 inserted single sentences, and short phrases in rater notes still quote the sources and are retained as experimental data.

SHA-256 of the files used in the experiments are in `data/docs_sha256.txt`.

The result files, logs and rater files in this package no longer contain the WHO or RFC chunk text: each chunk is replaced by its identifier and a SHA-256 prefix. The contexts can be restored from the original sources with scripts/fetch_corpus.py followed by scripts/restore_contexts.py, which checks every hash and writes a working copy; the judging scripts stop if a placeholder remains. The public release will follow the same rule. The Common Paper NDA text stays with attribution.

WHO does not endorse this work. The WHO fact sheets are cited as sources only.
