"""Stop judging scripts from running on withheld contexts."""
import sys

MARK = "chunk text withheld"


def assert_contexts_restored(rows):
    """rows: list of result rows with a 'context' dict. Exit with a clear message if any chunk text is a placeholder."""
    for r in rows:
        c = r.get("context")
        if isinstance(c, dict):
            for ch in c.get("chunks", []):
                if MARK in ch.get("text", ""):
                    sys.exit("The saved contexts contain withheld third-party text (placeholders). "
                             "Rebuild the corpus (scripts/fetch_corpus.py), then run scripts/restore_contexts.py "
                             "and re-run this script with RESULTS_DIR=results_restored. See README section B.")
