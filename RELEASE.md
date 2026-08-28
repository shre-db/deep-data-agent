# Release Notes

All notable changes to deep-data-agent are documented here.
Format: newest first, [Keep a Changelog](https://keepachangelog.com) style.
Versioning: [semantic versioning](https://semver.org) — the project stays on
`0.x` while in prototype stage; minor bumps mark feature releases.

## [v0.2.0] — 2026-08-28

First feature release after the `proto-v1` milestone.

### Added

- **Conversation memory**: SQLite checkpointer (`SqliteSaver`) persists every
  thread under `.checkpoints/threads.db`. Multi-turn follow-ups work within
  a session; `--thread-id <name>` resumes a conversation in a later process.
  CLI and web UI share the same thread store.
- **Auto-generated session threads**: runs without `--thread-id` start a
  fresh conversation with an id like `session-20260828-143005`, printed at
  startup so it can be resumed later.
- **Per-thread artifacts**: outputs are saved under `artifacts/<thread-id>/`
  (thread ids sanitized for path safety), so sessions don't overwrite each
  other.
- **Streamlit chat UI** (`uv run streamlit run app/ui.py`):
  - Chat with the agent in the browser; sidebar with dataset path, model
    display, and thread controls (new conversation / load existing thread).
  - Research trace (agent commentary, tool calls, tool results) rendered in
    a collapsible "Analysis trace" panel per message — not a streaming dump.
  - Charts render inline under the answer; CSV tables behind expanders;
    scripts listed as captions.
  - "Load thread" restores a previous conversation from the checkpointer,
    including reconstructed traces and existing artifacts.
- **Shared event stream** (`app/events.py`): one structured streaming path
  (`text_delta` / `tool_call` / `tool_result` / `final`) consumed by both
  the CLI and the web UI.
- **`MAX_TOKENS` setting**: caps requested completion tokens (default 8192)
  so providers with small credit balances (e.g. OpenRouter free tier) don't
  reject requests.

### Fixed

- The final answer no longer appears twice (once inside the trace, once as
  the response); streamed text is buffered per message id and the final
  message's buffer is suppressed.
- "Load thread" in the UI now actually restores the conversation history;
  unknown thread ids show a warning instead of failing silently.

### Verified

- Acceptance scenarios A (aggregation), B (visualization), C (data quality)
  end-to-end against Empero `glm-5.3-flash` and OpenRouter
  `z-ai/glm-5.3-flash`, via both CLI and UI code paths.
- Cross-process thread resume, multi-turn follow-ups, and thread restore
  tests (no LLM calls) — full suite: 8 tests.

### Known limitations (by design)

- Executed code is not process-isolated (env vars hidden, filesystem tools
  rooted at the project directory, but no real sandbox). Not for untrusted
  users.
- CSV ingestion only; single agent, no subagents; LangSmith tracing
  available but off by default.

## [proto-v1] — 2026-08-27

Pre-semver milestone tag marking the initial working prototype
(≈ `v0.1.0` semantically).

### What was built

- CLI natural-language data analysis agent:
  `uv run python -m app.main --data <csv> --question "<question>"`.
- `app/main.py` — CLI (single-command + interactive question), dataset
  report, live streaming trace of tool calls, tool results, and generated
  text; final answer + artifact listing.
- `app/agent.py` — `create_deep_agent` wiring; model factory for `empero:`,
  `openrouter:`, and `openai:` providers (OpenAI-compatible);
  `inspect_dataset` profiling tool; sandboxed `LocalShellBackend` rooted at
  the project directory with `inherit_env=False` and `MPLBACKEND=Agg`.
- `app/prompts.py` — analyst system prompt (plan-first, compute-don't-guess,
  validate, save artifacts) and required output format
  (Answer / Key findings / Caveats / Artifacts).
- `data/sample_sales.csv` — 438 x 8 synthetic sales dataset with an upward
  trend, channel/region variance, 4 duplicate rows, 12 missing values, and
  3 extreme revenue outliers (regenerable via `scripts/generate_sample_data.py`).
- `tests/test_smoke.py` — smoke tests (imports, dataset, agent construction,
  dataset tool); no LLM calls.

### Verified scenarios

- **A — Basic aggregation**: "What is total revenue by channel?" — channel
  totals cross-validated two ways; largest channel identified; no
  fabricated numbers.
- **B — Visualization**: "Show monthly revenue and explain the trend." —
  monthly grouping, outlier-adjusted trend line chart saved to `artifacts/`,
  evidence-backed explanation.
- **C — Data quality**: "Are there any data quality issues?" — found the 4
  duplicates, 12 missing values, and 3 outliers with counts, impact, and a
  flagged-rows CSV.

### Known limitations (at that time)

- No conversation memory across runs; one question per invocation.
- Executed code not process-isolated; not for untrusted users.
- CSV only; single agent, no subagents.
