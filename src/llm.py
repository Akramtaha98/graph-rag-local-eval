"""
Thin wrapper around a local Ollama server for text generation, plus a
sentence-transformers wrapper for embeddings. Both run entirely on your
laptop -- no API keys, no cloud calls.

Setup (one time):
    1. Install Ollama: https://ollama.com/download
    2. ollama pull llama3
    3. ollama serve   (usually starts automatically after install)
"""

import json
import time
import requests

from config import (
    OLLAMA_GENERATE_URL,
    OLLAMA_MODEL,
    EMBED_MODEL_NAME,
)

_embedder = None


def get_embedder():
    """Lazy-load the sentence-transformers model (loaded once, reused)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed_texts(texts):
    """Return a numpy array of embeddings, one row per input string."""
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def generate(prompt: str, temperature: float = 0.2, retries: int = 3, model: str = None) -> str:
    """
    Call the local Ollama /api/generate endpoint with a plain-text prompt
    and return the model's plain-text response.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"Could not reach Ollama at {OLLAMA_GENERATE_URL} after {retries} attempts. "
        f"Is `ollama serve` running and have you run `ollama pull {OLLAMA_MODEL}`? "
        f"Last error: {last_err}"
    )


def _try_parse_json(raw: str):
    """Best-effort extraction of a JSON object/array from a raw LLM response."""
    text = raw.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break

    start_candidates = [i for i in (text.find("["), text.find("{")) if i != -1]
    if start_candidates:
        start = min(start_candidates)
        text = text[start:]

    for end in range(len(text), 0, -1):
        candidate = text[:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def generate_json(prompt: str, temperature: float = 0.0, model: str = None, retries: int = 3) -> dict | list:
    """
    Call generate() and parse the response as JSON. Local LLMs sometimes wrap
    JSON in prose, code fences, or occasionally emit truncated/malformed JSON.
    This retries with an increasingly explicit "return ONLY valid JSON" nudge
    before giving up, so a single flaky generation doesn't silently zero out
    a downstream score (this previously caused a handful of 0.0 evaluator
    scores that were parsing failures, not real answer failures).
    """
    current_prompt = prompt
    for attempt in range(retries):
        raw = generate(current_prompt, temperature=temperature, model=model)
        parsed = _try_parse_json(raw)
        if parsed is not None and parsed != []:
            return parsed
        current_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON. "
              "Return ONLY the JSON object/array with no prose, no markdown fences, "
              "and no trailing commentary before or after it."
        )
    return []
