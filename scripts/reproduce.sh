#!/usr/bin/env bash
# One entry point for the whole study.
#
# Every stage is resumable, so this can be re-run after an interruption and will
# pick up where it stopped. Expect roughly a day of wall-clock on four CPU cores
# for the full ladder; see docs/compute.md for the breakdown and for the reason
# everything runs single-threaded.
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
export HF_HOME="${HF_HOME:-$PWD/.hf}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "== 0. sanity: the numerical path must be sound =="
$PY tests/test_core.py

echo "== 1. model weights (~14 GB, smallest first) =="
$PY scripts/download_models.py --ladder

echo "== 2. probe item pools =="
$PY scripts/build_items.py

echo "== 3. cached scoring == uncached scoring =="
$PY scripts/verify_scoring.py --model smollm2-135m --n 8

echo "== 4. Studies 1 and 2: the probe =="
$PY scripts/launch_probes.py --phases ABC

echo "== 5. Study 3: end-to-end rollouts =="
for m in smollm2-135m qwen2.5-0.5b llama3.2-1b; do
  $PY scripts/run_agent_study.py --models "$m" --env toolshed \
      --n-tasks 24 --max-steps 6 --max-new-tokens 32
done

echo "== 6. analysis, tables, figures, manuscript macros =="
$PY scripts/analyze_probe.py
$PY scripts/analyze_agent.py
$PY scripts/make_env_table.py
$PY scripts/dump_prompts.py
$PY scripts/make_qualitative.py
$PY scripts/make_facts.py
$PY scripts/make_figures.py
$PY scripts/fix_mangled_refs.py paper
$PY scripts/check_latex.py
$PY scripts/check_tables.py

echo "== 7. bibliography (re-verifies every citation against arXiv) =="
$PY scripts/fetch_citations.py

echo
echo "Done. Build the manuscript with:"
echo "  cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main"
