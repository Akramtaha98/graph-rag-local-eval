"""Records the model build, Ollama version and hardware used (reviewer comment B4). No model call is made.
Usage (from graph_rag_project/, Ollama running):  python3 scripts/env_report.py   -> results/env_report.txt"""
import json, os, platform, subprocess, sys
import requests
sys.path.insert(0, ".")
from config import OLLAMA_HOST, OLLAMA_MODEL, RESULTS_DIR


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(unavailable: {e})"


lines = []
add = lines.append
add(f"model tag requested: {OLLAMA_MODEL}")
add("ollama version: " + sh("ollama --version"))
try:
    tags = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=20).json().get("models", [])
    for m in tags:
        if m.get("name", "").split(":")[0] == OLLAMA_MODEL.split(":")[0]:
            add(f"installed: {m.get('name')} digest={m.get('digest')} size_bytes={m.get('size')} details={json.dumps(m.get('details'))}")
    show = requests.post(f"{OLLAMA_HOST}/api/show", json={"model": OLLAMA_MODEL}, timeout=30).json()
    add("show.details: " + json.dumps(show.get("details")))
    mi = show.get("model_info", {})
    for k in ("general.architecture", "general.parameter_count", "general.file_type", "llama.context_length"):
        if k in mi: add(f"{k}: {mi[k]}")
except Exception as e:  # noqa: BLE001
    add(f"(api query failed: {e})")
add("--- ollama show (cli) ---\n" + sh(f"ollama show {OLLAMA_MODEL}"))
add("--- hardware ---")
add("platform: " + platform.platform())
if sys.platform == "darwin":
    add("cpu: " + sh("sysctl -n machdep.cpu.brand_string"))
    add("logical cores: " + sh("sysctl -n hw.ncpu"))
    mem = sh("sysctl -n hw.memsize")
    add("memory: " + (f"{int(mem)/2**30:.0f} GiB" if mem.isdigit() else mem))
    add("macOS: " + sh("sw_vers -productVersion"))
else:
    add("cpu: " + sh("lscpu | grep 'Model name'"))
    add("memory: " + sh("free -h | head -2"))
add("python: " + sys.version.split()[0])
txt = "\n".join(lines)
open(f"{RESULTS_DIR}/env_report.txt", "w", encoding="utf-8").write(txt + "\n")
print(txt)
