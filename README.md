# deep-data-agent

Prototype of a natural-language data analysis agent built with
[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview/),
pandas, and matplotlib.

Give it a CSV and a question; it plans the analysis, writes and executes
pandas code, validates the numbers, generates plots, and returns an
evidence-backed answer with saved artifacts.

## Quickstart

```bash
git clone <repo>
cd deep-data-agent
uv sync
cp .env.example .env   # add an API key if your provider requires one
uv run python -m app.main \
  --data data/sample_sales.csv \
  --question "What are the main trends in revenue?"
```

Omit `--question` to be prompted interactively:

```bash
uv run python -m app.main --data data/sample_sales.csv
```

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

## Project layout

```
app/
├── main.py      # CLI: args, dataset report, agent invocation, artifact report
├── agent.py     # model factory, dataset tool, sandboxed backend, agent creation
└── prompts.py   # analyst system prompt + output format
data/            # sample synthetic sales dataset
artifacts/       # runtime-generated scripts, tables, plots (gitignored)
tests/           # smoke tests (no LLM calls)
scripts/         # sample data generator
```

## Testing

```bash
uv run pytest
```

## Security notes

This is a prototype. Executed code runs via `LocalShellBackend` with
`inherit_env=False` (secrets are not visible to analysis code) and a virtual
filesystem rooted at the project directory, but the shell itself is not
process-isolated. Do not expose it to untrusted users.

## Regenerating the sample data

```bash
uv run python scripts/generate_sample_data.py
```
