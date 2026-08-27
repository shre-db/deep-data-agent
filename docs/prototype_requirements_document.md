# Deep Agents Data Analysis Agent — Prototype Requirements

## 1. Objective

Build a working prototype of a **natural-language data analysis agent** using Python and Deep Agents.

The prototype should allow a user to provide a CSV dataset and ask questions such as:

- "What are the main trends in this dataset?"
- "Which category has the highest revenue?"
- "Are there any obvious outliers?"
- "Plot monthly revenue."
- "Compare conversion rates by channel."

The agent should autonomously:

1. Inspect the dataset.
2. Plan the analysis.
3. Write and execute Python analysis code.
4. Generate tables and visualizations where useful.
5. Validate its results.
6. Produce a concise natural-language answer with references to the generated artifacts.

This is explicitly a **prototype**, not a production system.

## 2. Time Constraint

Target implementation time: **≤ 60 minutes**.

Optimize for:

- working end-to-end flow;
- minimal dependencies;
- simple local execution;
- demonstrable agent behavior.

Do NOT spend time on:

- authentication;
- web UI;
- databases;
- multi-user support;
- deployment;
- sophisticated frontend;
- production-grade sandbox infrastructure;
- complex evaluation infrastructure;
- streaming UI;
- persistent memory.

## 3. Technology Choices

### Required

- Python 3.11+
- `deepagents`
- A tool-calling LLM supported by Deep Agents
- pandas
- matplotlib
- Python CLI

Use `uv` for environment/dependency management if available.

Deep Agents currently requires Python 3.11+ and supports tool-calling models from providers including OpenAI and Anthropic.

### Suggested model

Default to an environment-configurable model, for example:

`openai:gpt-5.5`

Do not hard-code the provider into the application architecture. The model should be configurable through an environment variable.

## 4. User Experience

The primary interface is a CLI.

Example:

```bash
uv run python -m app.main \
  --data data/sample_sales.csv \
  --question "What are the main trends in revenue?"
```

Expected behavior:

```text
Loading dataset...
Planning analysis...
Running analysis...
Generating visualization...

Answer:

Revenue increased steadily over the observed period...
The strongest growth occurred in...
The largest category was...

Artifacts:
- artifacts/revenue_trend.png
- artifacts/analysis.py
- artifacts/summary.md
```

A second supported interaction should allow an interactive question:

```bash
uv run python -m app.main --data data/sample_sales.csv
```

Then:

```text
Data loaded: 10,000 rows × 8 columns

Question: Which channel has the highest conversion rate?
```

Interactive mode is optional if time becomes constrained. The single-command mode is mandatory.

## 5. Functional Requirements

### FR-1: Dataset ingestion

The application must accept a local CSV file.

Requirements:

- verify the file exists;
- load it with pandas;
- report row count and column count;
- expose column names and basic dtypes to the agent;
- gracefully report malformed/unreadable CSV files.

Example:

```text
Dataset: data/sample_sales.csv
Rows: 10,000
Columns: 8

Columns:
- date: datetime/string
- channel: object
- revenue: float
- orders: integer
...
```

Do not attempt to support arbitrary file formats in the prototype.

### FR-2: Natural-language analysis

The user supplies a natural-language question.

The agent must decide what analysis is appropriate rather than requiring the user to specify pandas operations.

Examples:

```text
"What are the main trends?"
"Find unusual values."
"Which product category performs best?"
"Show revenue by month."
```

### FR-3: Agent planning

Use Deep Agents' built-in planning capabilities rather than implementing a custom planner.

The agent should create an explicit analysis plan before performing substantial work.

The plan should generally contain:

1. Understand the dataset.
2. Determine relevant columns.
3. Perform appropriate analysis.
4. Validate calculations.
5. Generate visualizations if useful.
6. Summarize findings.

Deep Agents is specifically designed around planning and multi-step work, so the prototype should demonstrate this capability rather than hiding all logic inside one giant tool.

### FR-4: Code execution

The agent must be able to execute Python code for analysis.

The execution environment should provide:

- pandas;
- numpy;
- matplotlib;
- the input dataset;
- an artifacts directory.

Prefer Deep Agents' sandbox/backend mechanisms rather than giving the LLM uncontrolled access to the developer's host environment.

For the prototype, a local/sandboxed execution approach is acceptable.

### FR-5: Analysis operations

The prototype should support common exploratory analysis:

- dataset dimensions;
- missing values;
- descriptive statistics;
- unique-value counts;
- group-by aggregations;
- sorting/ranking;
- correlations where appropriate;
- time-series aggregation;
- outlier detection;
- basic comparisons.

Do not build a bespoke tool for every operation.

The agent should generate Python/pandas code dynamically.

### FR-6: Visualization

The agent should be able to generate plots when useful.

Support at minimum:

- line chart;
- bar chart;
- histogram;
- scatter plot.

Plots should be saved under:

```text
artifacts/
```

The final response should mention generated plot filenames.

### FR-7: Result validation

Before producing the final answer, the agent should verify important numerical claims.

For example:

```text
Calculated total revenue: $1,284,392
Top channel: Paid Search
Paid Search revenue: $421,812
```

The agent should derive these values from executed code rather than guessing from the dataset description.

### FR-8: Final response

The final answer should contain:

1. Direct answer to the user's question.
2. 2–5 important findings.
3. Relevant caveats.
4. Generated artifacts, if any.

Example:

```text
## Answer

Paid Search has the highest revenue at $421,812.

### Key findings

- Paid Search contributes 32.8% of total revenue.
- Revenue increased 14% from January to June.
- Mobile traffic has a lower conversion rate than desktop.

### Artifacts

- artifacts/revenue_by_channel.png
- artifacts/monthly_revenue.png
- artifacts/analysis.py
```

The agent should avoid presenting unsupported causal claims.

## 6. Agent Architecture

Use one primary Deep Agent.

Conceptually:

```text
User
  |
  v
Deep Analysis Agent
  |
  +--> inspect dataset
  |
  +--> create analysis plan
  |
  +--> write Python analysis
  |
  +--> execute Python
  |
  +--> inspect results
  |
  +--> generate plots
  |
  +--> validate findings
  |
  v
Final answer + artifacts
```

Do not introduce multiple subagents initially.

However, structure the code so a specialized analyst subagent could be added later.

## 7. Deep Agent Configuration

Use `create_deep_agent`.

The system prompt should establish that the agent is:

- a careful data analyst;
- evidence-driven;
- allowed to inspect and analyze files;
- expected to execute code rather than mentally calculate results;
- expected to validate important findings;
- expected to create visualizations when they materially improve understanding;
- expected to explain uncertainty;
- prohibited from fabricating values.

The agent should have access to:

- dataset inspection;
- Python execution;
- filesystem/artifact operations.

Deep Agents already bundles filesystem and context-management capabilities, which makes it a good fit for keeping generated analysis scripts and intermediate artifacts out of the model's conversational context.

## 8. Suggested System Prompt

Use approximately this behavior:

```text
You are a careful data analysis agent.

Your job is to answer the user's data-analysis question using the supplied dataset.

Always:
1. Inspect the dataset before analyzing it.
2. Create a short plan for the analysis.
3. Use Python/pandas for numerical calculations.
4. Never invent or estimate numerical results when they can be calculated.
5. Validate important calculations before reporting them.
6. Generate a visualization when it materially helps answer the question.
7. Save useful scripts, tables, and plots under the artifacts directory.
8. Distinguish correlation from causation.
9. Call out missing data, small samples, or other important limitations.
10. Give the user a concise final answer supported by the analysis.

Prefer simple, reproducible Python code.
```

## 9. Repository Structure

Start with this structure:

```text
deep-data-agent/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── agent.py
│   └── prompts.py
│
├── data/
│   └── sample_sales.csv
│
├── artifacts/
│   └── .gitkeep
│
└── tests/
    └── test_smoke.py
```

Keep it intentionally small.

### `app/main.py`

Responsibilities:

- parse CLI arguments;
- validate dataset path;
- create the agent;
- invoke the agent;
- print the final response;
- report generated artifacts.

### `app/agent.py`

Responsibilities:

- construct the Deep Agent;
- configure model;
- configure analysis tools/backend;
- configure filesystem/artifact behavior.

Example conceptual structure:

```python
from deepagents import create_deep_agent

def create_analysis_agent(model):
    return create_deep_agent(
        model=model,
        tools=[...],
        system_prompt=ANALYST_PROMPT,
    )
```

### `app/prompts.py`

Store:

- system prompt;
- analysis instructions;
- output-format instructions.

Keeping prompts separate makes iteration easy.

### `data/sample_sales.csv`

Include a small synthetic dataset so the prototype can be demonstrated without external data.

Suggested columns:

```text
date
channel
product
region
orders
revenue
customers
conversion_rate
```

Generate approximately 100–500 rows.

### `artifacts/`

Runtime-generated:

```text
analysis.py
summary.md
*.png
*.csv
```

Do not commit generated artifacts by default.

### `tests/test_smoke.py`

At minimum, verify:

- application imports;
- sample dataset loads;
- agent construction succeeds.

A full LLM integration test is optional because it may require API credentials and incur cost.

## 10. Dependencies

Start with the minimum:

```text
deepagents
pandas
numpy
matplotlib
python-dotenv
```

Add the provider-specific LangChain integration only if required by the selected model.

Do not add:

- FastAPI;
- Streamlit;
- Jupyter;
- SQLAlchemy;
- Redis;
- PostgreSQL;
- Docker;

unless they become necessary.

## 11. Configuration

`.env.example`:

```text
MODEL=openai:gpt-5.5
OPENAI_API_KEY=
```

The application should fail with a clear error if the selected model requires an API key that is missing.

Optional:

```text
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=deep-data-agent
```

Deep Agents supports native LangSmith tracing through its LangGraph foundation, so tracing can be enabled later without changing the core architecture.

## 12. Security Requirements

This is a prototype, but code execution is the highest-risk component.

The agent must NOT receive unrestricted access to the developer's filesystem.

At minimum:

- restrict analysis files to the project/workspace;
- restrict generated artifacts to `artifacts/`;
- avoid exposing secrets/environment variables to executed analysis code;
- do not allow arbitrary network access from analysis code unless explicitly enabled;
- never execute code supplied directly by an untrusted external user without an appropriate sandbox.

Deep Agents explicitly follows a "trust the LLM" model and recommends enforcing boundaries at the tool/sandbox level rather than relying on the model to self-police.

## 13. Non-Functional Requirements

### Simplicity

A developer should be able to understand the entire prototype in under 15 minutes.

### Reproducibility

The same dataset and question should generally produce equivalent numerical results.

### Observability

It should be possible to inspect:

- generated Python code;
- generated plots;
- intermediate output;
- final answer.

### Performance

A normal analysis should complete within a few minutes, depending primarily on LLM latency.

Do not optimize prematurely.

## 14. Acceptance Criteria

The prototype is complete when all of the following work:

### Scenario A — Basic aggregation

Input:

```text
"What is total revenue by channel?"
```

Expected:

- agent inspects data;
- executes pandas code;
- returns channel-level totals;
- identifies the largest channel;
- no fabricated numbers.

### Scenario B — Visualization

Input:

```text
"Show monthly revenue and explain the trend."
```

Expected:

- agent groups revenue by month;
- generates a line chart;
- saves the chart under `artifacts/`;
- describes the trend using calculated values.

### Scenario C — Data quality

Input:

```text
"Are there any data quality issues?"
```

Expected:

- agent checks missing values;
- checks obvious duplicates/inconsistencies;
- reports findings with counts.

### Scenario D — Exploratory question

Input:

```text
"What are the three most interesting things about this dataset?"
```

Expected:

- agent explores multiple dimensions;
- identifies 3 evidence-backed findings;
- optionally generates supporting visualizations.

## 15. Explicitly Out of Scope

Do not implement during the one-hour prototype:

- authentication;
- web UI;
- user accounts;
- persistent conversations;
- database ingestion;
- Excel/Parquet support;
- cloud deployment;
- enterprise security;
- sophisticated permissions UI;
- automated email/Slack delivery;
- multi-agent orchestration;
- long-term memory;
- custom LangGraph workflows;
- production evaluation framework.

These can be Phase 2.

## 16. Suggested 60-Minute Build Plan

### 0–10 minutes

- Create repository.
- Initialize `uv`.
- Install dependencies.
- Configure API key.
- Verify a trivial Deep Agent works.

### 10–20 minutes

- Create `app/agent.py`.
- Add analyst system prompt.
- Add dataset/file context.
- Add Python execution capability.

### 20–35 minutes

- Implement CLI.
- Load sample CSV.
- Run first real analysis.
- Make sure generated Python can be inspected.

### 35–45 minutes

- Add artifact generation.
- Add visualization support.
- Save plots and analysis scripts.

### 45–55 minutes

- Add validation behavior.
- Add smoke test.
- Test three representative questions.

### 55–60 minutes

- Clean README.
- Add `.env.example`.
- Confirm one-command startup.
- Remove unnecessary dependencies/code.

## 17. Definition of Done

A fresh developer should be able to run:

```bash
git clone <repo>
cd deep-data-agent
uv sync
cp .env.example .env
# add API key
uv run python -m app.main \
  --data data/sample_sales.csv \
  --question "What are the main trends in revenue?"
```

and receive:

1. A natural-language answer.
2. Evidence-backed numerical findings.
3. At least one generated analysis artifact.
4. A reproducible analysis script.
5. No manual pandas coding by the user.

## 18. Future Architecture

If the prototype proves useful, evolve toward:

```text
                    ┌──────────────────┐
                    │    User / UI     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Analysis Agent  │
                    └────────┬─────────┘
                             │
             ┌───────────────┼────────────────┐
             │               │                │
      ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
      │ Data Profiler│ │   Analyst   │ │  Reviewer   │
      └─────────────┘ └─────────────┘ └─────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Sandbox / Python │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ Artifact Storage │
                    └──────────────────┘
```

Potential Phase 2 additions:

- Excel/Parquet/SQL;
- richer visualization;
- analyst/reviewer subagents;
- persistent sessions;
- LangSmith evaluation;
- approval gates for expensive or risky operations;
- web interface;
- remote sandbox execution.

The current Deep Agents architecture is well suited to this evolution because it supports subagents, pluggable filesystem backends, sandboxes, human-in-the-loop controls, and persistent state.

## 19. Reference Documentation

Use the current Deep Agents documentation rather than relying on older examples:

- Deep Agents overview
- Deep Agents quickstart
- Deep Agents data-analysis guide
- Deep Agents Python API reference

The official data-analysis guide specifically demonstrates the pattern of accepting a CSV, performing exploratory analysis, generating visualizations, and producing results, making it a useful implementation reference for this prototype.