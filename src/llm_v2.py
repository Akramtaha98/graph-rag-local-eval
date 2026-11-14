"""Ollama wrapper for the v2 re-run. Differences from src/llm.py:
  * sets num_ctx explicitly (default 8192) so long prompts are not silently truncated by Ollama;
  * returns Ollama's prompt_eval_count so you can verify no prompt exceeded the window;
  * no JSON-repair guessing: callers parse strictly.
Put this file next to config.py in graph_rag_project/ (or in src/) and import from there."""
import time
import requests
from config import OLLAMA_GENERATE_URL, OLLAMA_MODEL

NUM_CTX = 8192


def generate_with_stats(prompt, temperature=0.0, retries=3, model=None, num_ctx=NUM_CTX):
    payload = {"model": model or OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature, "num_ctx": num_ctx}}
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
            r.raise_for_status()
            d = r.json()
            return d.get("response", "").strip(), int(d.get("prompt_eval_count", -1))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed: {last}")


def generate(prompt, temperature=0.0, **kw):
    return generate_with_stats(prompt, temperature, **kw)[0]
