# Release Notes

All notable changes to deep-data-agent are documented here.
Format: newest first, [Keep a Changelog](https://keepachangelog.com) style.
Versioning: [semantic versioning](https://semver.org) — the project stays on
`0.x` while in prototype stage; minor bumps mark feature releases.

## [v0.3.0] — 2026-08-29

The UI polish release: structured traces, interactive charts, inline
artifacts, custom fonts, and the post-prototype product direction.

### Added

- **Interactive Plotly charts**: chart helpers (`app/agent_tools/charts.py`)
  save figure JSON with one consistent style — validated light/dark
  categorical palette, computed margins, horizontal legend — and the UI
  renders them theme-aware with `st.plotly_chart`. `plotly` added to
  dependencies; matplotlib remains as the fallback.
- **Inline artifact references**: a `![caption](artifacts/<thread>/<name>.json)`
  line (or a plain link, e.g. for `.csv` tables) renders the artifact at the
  point in the answer where it is discussed. Unreferenced artifacts collapse
  into a single "Artifacts (N)" gallery; code files (`.py`/`.sql`/`.sh`)
  render as syntax-highlighted code blocks.
- **Markdown preprocessing** (`app/markdown_utils.py`): currency amounts
  (`$3,643,063.54`, `$98–106`) stay literal while real LaTeX (`$x^2$`,
  `$$…$$`) still renders as math; a leading "Answer" heading is stripped.
- **Custom fonts**: Inter for UI text via native `[theme] font` config
  (`.streamlit/config.toml`) and JetBrains Mono for code, tool calls, trace
  metadata, JSON, and stack traces.
- **Answer framework**: the rigid Answer / Key findings / Caveats /
  Artifacts template is replaced by principles — answer first, numbers as
  evidence, mandatory limitations check — with the shape left to the
  question; the prose artifacts list is dropped (the UI shows them).
- **Product design doc**: `docs/analyst_on_staff.md` (+ designed HTML
  rendering) — thesis and core loop, six design principles, architecture,
  data model, canonical morning-briefing scenario, M0–M4 roadmap.

### Fixed

- **Duplicated tool output in the trace**: LangGraph's messages stream also
  emits node outputs (ToolMessages); only chat-model messages now enter the
  commentary buffer, so tool results no longer render twice.
- **Failed-command tracebacks** render inside their expander; the stray
  outside failure notice is removed; orphan output renders collapsed.
- **Chart labels clipped out of frame**: `automargin` plus margins computed
  from axis-title/legend presence.
- **LLM retries** (`LLM_MAX_RETRIES`, default 4) for transient TLS/connection
  drops on hosted endpoints.
- The UI persists partial turns when an analysis fails mid-run.

### Verified

- Full offline suite: 58 tests — event stream shape/dedupe/echo suppression,
  chart helper output, currency-vs-LaTeX preprocessing, answer segmentation,
  step grouping.
- Live-run review of the trace, inline charts/tables, fonts, and currency
  rendering (light and dark themes).

### Known limitations (by design)

- No process isolation (env vars hidden, filesystem rooted at the project
  directory, but no real sandbox); not for untrusted users.
- CSV-only connectors; single agent, no subagents.
- Plotly charts render in an iframe, so Inter inside chart labels depends on
  iframe font loading; fallback stacks keep charts clean either way.

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
