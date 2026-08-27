# Release: proto-v1

First tagged milestone of the deep-data-agent prototype.

## What was built

A CLI natural-language data analysis agent:

```bash
uv run python -m app.main --data data/sample_sales.csv --question "..."
```

The agent autonomously inspects the dataset, plans, writes and executes
pandas/matplotlib code, validates numbers, and returns a structured answer
with saved artifacts.

### Components

- `app/main.py` — CLI (single-command + interactive question), dataset
  report, **live streaming trace** of the agent's tool calls, tool results,
  and generated text; final answer + artifact listing.
- `app/agent.py` — `create_deep_agent` wiring; model factory for
  `empero:`, `openrouter:`, and `openai:` providers (OpenAI-compatible);
  `inspect_dataset` profiling tool; sandboxed `LocalShellBackend` rooted at
  the project directory with `inherit_env=False` (executed code cannot see
  host secrets) and `MPLBACKEND=Agg`.
- `app/prompts.py` — analyst system prompt (plan-first, compute-don't-guess,
  validate, save artifacts) and required output format
  (Answer / Key findings / Caveats / Artifacts).
- `data/sample_sales.csv` — 438 x 8 synthetic sales dataset with an upward
  trend, channel/region variance, 4 duplicate rows, 12 missing values, and
  3 extreme revenue outliers (regenerable via `scripts/generate_sample_data.py`).
- `tests/test_smoke.py` — 4 smoke tests (imports, dataset, agent construction,
  dataset tool); no LLM calls.

## Verified scenarios

Tested end-to-end against Empero `glm-5.3-flash` and OpenRouter
`z-ai/glm-5.3-flash`:

- **A — Basic aggregation**: "What is total revenue by channel?" — channel
  totals cross-validated two ways; largest channel identified; no
  fabricated numbers.
- **B — Visualization**: "Show monthly revenue and explain the trend." —
  monthly grouping, outlier-adjusted trend line chart saved to `artifacts/`,
  evidence-backed explanation.
- **C — Data quality**: "Are there any data quality issues?" — found the 4
  duplicates, 12 missing values, and 3 outliers with counts, impact, and a
  flagged-rows CSV.

## Known limitations (by design)

- No conversation memory across runs; one question per invocation.
- Executed code is not process-isolated (shell access; env vars hidden,
  filesystem tools rooted at the project, but no real sandbox). Not for
  untrusted users.
- CSV only; single agent, no subagents; no LangSmith tracing enabled by
  default.

## Next candidates

- Multi-turn sessions (checkpointer / interactive loop) for follow-up
  questions.
- Optional analyst/reviewer subagent.
