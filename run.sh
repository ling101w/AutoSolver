#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate autosolver-agent
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set." >&2
  echo "Run this first, then rerun ./run.sh:" >&2
  echo "  export OPENAI_API_KEY='your-api-key'" >&2
  exit 1
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://nasyh.cyou/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export AUTOSOLVER_LLM_MODEL="${AUTOSOLVER_LLM_MODEL:-${OPENAI_MODEL:-gpt-5.5}}"
export OPENAI_MODEL="$AUTOSOLVER_LLM_MODEL"

cases=("$@")
if [[ ${#cases[@]} -eq 0 ]]; then
  cases=(examples/demo_case.txt)
fi

mkdir -p runs/manual

python -m autosolver_agent.cli \
  --cases "${cases[@]}" \
  --llm-model "$AUTOSOLVER_LLM_MODEL" \
  --llm-base-url "$OPENAI_BASE_URL" \
  --out "${AUTOSOLVER_OUT:-runs/manual/generated_submit_solution.py}" \
  --budget "${AUTOSOLVER_BUDGET:-3600}" \
  --iterations "${AUTOSOLVER_ITERATIONS:-100}" \
  --strategy-workers "${AUTOSOLVER_STRATEGY_WORKERS:-1}" \
  --per-case-timeout "${AUTOSOLVER_PER_CASE_TIMEOUT:-10}" \
  --search-per-case-timeout "${AUTOSOLVER_SEARCH_PER_CASE_TIMEOUT:-10}" \
  --memory-dir "${AUTOSOLVER_MEMORY_DIR:-runs/autosolver_memory}" \
  --artifact-dir "${AUTOSOLVER_ARTIFACT_DIR:-runs/autosolver_artifacts}" \
  --summary-out "${AUTOSOLVER_SUMMARY_OUT:-runs/manual/summary.json}"
