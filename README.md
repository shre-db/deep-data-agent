# deep-data-agent

Prototype of a natural-language data analysis agent built with
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview/),
pandas, and matplotlib.

Give it a CSV and a question; it plans the analysis, writes and executes
pandas code, validates the numbers, generates plots, and returns an
evidence-backed answer with saved artifacts.

## Requirements

- Python 3.11+ (3.12 recommended)
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An API key for your chosen model provider (the default Empero endpoint is
  currently free and needs no key)

## Installation

```bash
git clone https://github.com/shre-db/deep-data-agent.git
cd deep-data-agent

# 1. Install dependencies (creates .venv automatically)
uv sync

# 2. Configure the environment
cp .env.example .env
# Edit .env: set MODEL and the matching API key (if required)

# 3. Sanity-check the setup (no LLM calls)
uv run pytest
```

## Usage

Single question (non-interactive):

```bash
uv run python -m app.main \
  --data data/sample_sales.csv \
  --question "What are the main trends in revenue?"
```

Interactive session — type questions one at a time; each run starts a fresh
thread (conversation remembered within it), so follow-ups like *"now break
that down by region"* work. Exit with `quit`, `exit`, `q`, or Ctrl-D:

```bash
uv run python -m app.main --data data/sample_sales.csv
```

Named threads — resume a previous conversation in a later process:

```bash
uv run python -m app.main --data data/sample_sales.csv --thread-id sales-2025
```

CLI options:

| Flag | Description |
|------|-------------|
| `--data PATH` | (required) Path to the CSV file to analyze |
| `--question TEXT` | Ask once and exit; omit for interactive mode |
| `--thread-id NAME` | Resume a named conversation thread. If omitted, a new thread is created per run with an auto-generated id (printed at startup, e.g. `session-20260828-143005`) |

Each conversation gets its own thread and its own artifact folder:
`artifacts/<thread-id>/`. To continue a previous conversation (with its
memory and artifacts), pass its `--thread-id` again.

## Example session

```text
Loading dataset...

Dataset: data/sample_sales.csv
Rows: 438
Columns: 8
...
Question: What is total revenue by channel?
Running analysis... (live trace below)

  [tool call] inspect_dataset({'path': 'data/sample_sales.csv'})
  [tool call] execute({'command': 'python analysis.py'})
  ...

============================================================

## Answer

Organic leads with $1,387,240.56 — 38% of total revenue...
...

Artifacts saved:
- artifacts/default/revenue_by_channel.png
- artifacts/default/analysis.py
```

## Web UI

A minimal Streamlit chat frontend is included:

```bash
uv run streamlit run app/ui.py
```

- Chat with the agent in the browser; each conversation gets its own thread
  and artifact folder (`artifacts/<thread>/`).
- The agent's research trace (commentary, tool calls, tool results) is shown
  in a collapsible "Analysis trace" panel per message.
- Charts and tables generated during a turn render inline under the answer;
  scripts are listed as plain captions.
- The sidebar has the dataset path, model display, and thread controls
  (new conversation / resume by thread id).

## Conversation memory

Sessions are persisted per thread with a SQLite checkpointer in
`.checkpoints/threads.db` (gitignored). Every run without `--thread-id`
starts a new thread; all questions in an interactive session share that
thread. Reusing `--thread-id <name>` in a later process restores that
conversation, including the model's knowledge of prior questions and
artifacts it created.

## Model configuration

The model is selected with the `MODEL` env var (`provider:model`):

| Provider   | Example value                     | API key env         |
|------------|-----------------------------------|---------------------|
| Empero     | `empero:glm-5.3-flash` (default)  | `EMPERO_API_KEY` (not currently required for the free endpoint) |
| OpenRouter | `openrouter:z-ai/glm-5.3-flash`   | `OPENROUTER_API_KEY` |
| OpenAI     | `openai:gpt-4o-mini`              | `OPENAI_API_KEY` |

## What it does

1. Loads and profiles the dataset (rows, columns, dtypes, missing values).
2. Plans the analysis using Deep Agents' built-in planning.
3. Writes and executes pandas/matplotlib code through a sandboxed shell
   backend rooted at the project directory (no access to your environment
   variables or secrets).
4. Saves scripts, tables, and plots under `artifacts/`.
5. Validates important numbers and returns a structured answer
   (Answer / Key findings / Caveats / Artifacts).

While it works, the CLI streams a live trace of the agent's tool calls,
tool results, and generated text — no blank-screen waiting.

## Project layout

```
app/
├── main.py      # CLI: args, dataset report, live trace, interactive loop, artifacts
├── agent.py     # model factory, checkpointer, dataset tool, sandboxed backend
├── events.py    # shared agent event stream (used by CLI and UI)
├── prompts.py   # analyst system prompt + output format
└── ui.py        # Streamlit chat frontend
data/            # sample synthetic sales dataset
artifacts/       # runtime-generated scripts, tables, plots, per thread (gitignored)
.checkpoints/    # SQLite conversation memory per thread (gitignored)
tests/           # smoke tests (no LLM calls)
scripts/         # sample data generator
docs/            # requirements + release notes
```

## Testing

```bash
uv run pytest          # smoke tests, no API key or LLM calls needed
```

## Cleaning up

```bash
rm -rf artifacts/* .checkpoints   # remove generated files and session memory
uv run python scripts/generate_sample_data.py   # regenerate the sample dataset
```

Tip: if a provider rejects requests with a token-budget error (e.g. small
OpenRouter credit balance), set `MAX_TOKENS=8192` in `.env` to cap the
requested completion size.

## Security notes

This is a prototype. Executed code runs via `LocalShellBackend` with
`inherit_env=False` (secrets are not visible to analysis code) and a virtual
filesystem rooted at the project directory, but the shell itself is not
process-isolated. Do not expose it to untrusted users.
