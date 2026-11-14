"""
Load raw documents (.txt, .md, .pdf) from DOCS_DIR and split them into
overlapping word-window chunks with stable IDs and source provenance.
"""

import os
import glob
import hashlib

from config import DOCS_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, EXCLUDE_DOCS


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(docs_dir: str = DOCS_DIR) -> list[dict]:
    """Return [{"source": filename, "text": full_text}, ...] for every supported file."""
    documents = []
    patterns = ["*.txt", "*.md", "*.pdf"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(docs_dir, "**", pattern), recursive=True))

    if not paths:
        raise FileNotFoundError(
            f"No .txt, .md, or .pdf files found in {docs_dir}. "
            f"Drop your domain-specific documents there before running `build`."
        )

    for path in sorted(paths):
        if os.path.basename(path) in EXCLUDE_DOCS:
            print(f"[ingest] Skipping {path} (listed in EXCLUDE_DOCS in config.py).")
            continue
        try:
            text = _read_pdf(path) if path.lower().endswith(".pdf") else _read_txt(path)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] Skipping {path}: {e}")
            continue
        text = text.strip()
        if text:
            documents.append({"source": os.path.relpath(path, docs_dir), "text": text})
    return documents


def chunk_text(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Sliding-window chunking over whitespace-tokenized words."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs_dir: str = DOCS_DIR) -> list[dict]:
    """
    Return a flat list of chunk records:
        {"chunk_id": str, "source": str, "text": str}
    chunk_id is a short stable hash so re-runs are reproducible.
    """
    documents = load_documents(docs_dir)
    all_chunks = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            raw_id = f"{doc['source']}::{i}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
            all_chunks.append({"chunk_id": chunk_id, "source": doc["source"], "text": chunk})
    print(f"[ingest] Loaded {len(documents)} document(s) -> {len(all_chunks)} chunk(s).")
    return all_chunks
