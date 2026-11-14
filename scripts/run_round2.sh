#!/bin/bash
# Round-2 measurements (reviewer comments A6, A10, A11, B4). Run from graph_rag_project/ with Ollama serving.
# Keep the laptop plugged in and otherwise idle; the timing step is sensitive to other load.
set -e
python3 scripts/env_report.py                       # B4: model digest, quantization, Ollama version, hardware (seconds)
python3 scripts/token_audit.py                      # A6: maximum prompt tokens per prompt type, cache-proof (about 10 min)
python3 scripts/timing_run.py                       # A10: indexing and per-query latency (about 1 h)
ollama pull qwen2.5:7b                              # A11: a second model family (4.7 GB download)
python3 scripts/judge_other_model.py qwen2.5:7b     # A11: second judge on the 60 validation answers (about 30-60 min)
python3 scripts/analysis_other.py results/judge_other_qwen2_5_7b_details.json results
echo "ALL DONE. Send me results/env_report.txt, token_audit.json, timing.json, judge_other_qwen2_5_7b_details.json and the last printout."
