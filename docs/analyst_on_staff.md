# The Analyst on Staff

Design doc for evolving the deep-data-agent from a chat window into an autonomous data analyst: scheduled briefings, business definitions, lineage, and audience-aware delivery.

Status: draft for discussion — not yet scoped into milestones.
A designed, shareable rendering of this doc exists at the published artifact link (see README notes when added).

---

## 1. Thesis

Today this project is a chat window that answers questions. The next version is an **analyst on staff**: it monitors data on a schedule, knows the business's definitions and history, runs validated pipelines, and delivers briefings and actions to the right people.

The unit of value is not the answer — it is the closed loop:

```
data → analyst core → validated run → deliverable → audience → feedback → memory ┐
        ↑_____________________________________________________________________┘
```

The chat window is demoted but not discarded: it becomes the **console** — the place where runs are inspected, definitions are corrected, and briefings are reviewed before they go out.

## 2. The core loop

- **Data** — connectors feed the agent (CSV today; databases and APIs later), with freshness checks that raise "data is stale" before analysis starts.
- **Analyst core** — the Deep Agent: plans, computes with code, validates, interprets, narrates.
- **Validated run** — vetted scripts plus lineage metadata. Every number in the output traces to dataset, script version, and run timestamp. Re-runs diff against history.
- **Deliverable** — a briefing, table, or dashboard rendered for a specific audience (a VP gets five minutes of narrative and decisions; an analyst gets methodology, caveats, and code).
- **Audience** — a person with a stored profile: role, expertise level, delivery channel.
- **Feedback** — corrections and ratings from the audience become memory.
- **Memory** — business definitions, historical baselines, and past decisions. The compounding asset: every correction makes every future run smarter.

## 3. Design principles

1. **Deterministic plumbing, LLM brains.** Scheduling, connectors, storage, and delivery are plain code. The model interprets and narrates; it never sits in the correctness-critical path of computing a number.
2. **Analyses as code.** Every repeatable analysis is a versioned, vetted script. Edits are reviewed; re-runs diff against history. A churn number is computed by an approved pipeline, not regenerated from scratch each morning.
3. **Lineage everywhere.** Every number in a deliverable cites its dataset, script version, and run. No orphan statistics.
4. **Audience is a rendering policy.** Expertise level is a property of the deliverable, not something inferred from conversation. One analysis, N renderings.
5. **Feedback is memory.** Verbal reinforcement learning in the pragmatic sense: corrections and ratings become definition updates and few-shot context for future runs — inspectable, no fine-tuning.
6. **Actions are gated.** Creating tickets and sending alerts start behind human approval. The trust bar for acting on someone's behalf is higher than the bar for informing them.

## 4. Architecture

| Component | Responsibility | State today |
|---|---|---|
| Connectors | Ingest datasets, check freshness | CSV files, ad hoc |
| Knowledge layer | Business definitions, baselines, feedback log | Thread-scoped only |
| Analyst core | Deep Agent: plan → compute → validate → narrate | Exists (`app/agent.py`) |
| Scheduler | Daily/weekly runs, each run = one thread | Manual chat only |
| Run store | Artifacts + lineage sidecars, versioned | Artifacts per thread |
| Delivery | Render briefings per audience; send to Slack/email | Chat rendering only |
| Action layer | Investigation tasks, alerts with severity + approval | None |
| Console | Inspect runs, correct definitions, review briefings | Exists (`app/ui.py`) |

## 5. Data model

| Entity | Key fields | Notes |
|---|---|---|
| `business_definition` | name, definition, owner, version, supersedes | "When I say churn, I mean X." Versioned; corrections append, never silently overwrite |
| `audience_profile` | person, role, expertise, channel, preferences | Asked once per recipient, stored |
| `run` | id, trigger (schedule/manual), status, lineage refs | A thread + its metadata |
| `artifact` | path, kind, source_data_hash, script_version, run_id | The lineage sidecar |
| `briefing` | run_id, audience, rendered content, delivery status | One run renders N briefings |
| `feedback` | briefing_id, kind (correction/rating), text | Corrections may spawn definition updates |
| `task` | created_from, assignee, severity, status | The action layer's output |

## 6. Canonical scenario — the morning briefing

1. **07:00** — the scheduler starts the daily run: product, billing, and CRM data, each freshness-checked.
2. **Analyst** — the core runs the vetted pipelines for activation, retention, and revenue; diffs against historical baselines.
3. **Anomaly** — retention in the SMB segment shows an accelerating decline. The deterministic baseline flags it; the model attributes likely causes: a billing API error spike and a pricing change on Tuesday.
4. **Validation** — cross-check tests pass; lineage is attached to every cited number.
5. **Rendering** — the briefing renders at two altitudes: a five-minute narrative for the VP Product (one chart per finding, decisions needed) and a full version for the analyst (methodology, caveats, scripts).
6. **Delivery** — the VP's briefing lands in Slack with a review gate; the analyst's version lands in their inbox.
7. **Escalation** — churn severity crosses threshold: the system proposes an investigation task (assignee, severity, lineage links) for human approval.
8. **Feedback** — the VP replies "by churn I meant paid-only accounts." The correction becomes a new version of the `churn` definition, and tomorrow's run uses it.

## 7. Roadmap

| Milestone | Scope | Success criteria |
|---|---|---|
| **M0 — Chat prototype** | Today's app | (done) |
| **M1 — Briefing + knowledge** | Briefing run type, definitions store, audience profiles | One daily briefing renders at two altitudes from stored definitions |
| **M2 — Schedule + delivery** | Scheduler, Slack/email renderer, human review gate | Briefings arrive without opening the app; console used for review |
| **M3 — Anomaly baselines** | Deterministic stats (seasonality, z-scores) feeding findings | Anomalies are cited against historical patterns with confidence |
| **M4 — Action layer** | Investigation tasks, severity routing, approval flows | Escalation creates a tracked task with a lineage link |

## 8. Mapping to today's codebase

- `app/agent.py` — Deep Agent + `LocalShellBackend` + the `analysis.py` overwrite pattern → the analyst core and the seed of analyses-as-code.
- `app/events.py` — the structured event trace → the run record briefings render from.
- `app/agent_tools/charts.py` — the standardized chart language → the visual layer of briefings.
- `app/prompts.py` — the answer framework → the analyst's job description; audience profile and definitions become injected context.
- Checkpointer + `artifacts/<thread>/` → the run store; lineage arrives as metadata sidecars per artifact.
- `app/ui.py` — the console: inspect runs, correct definitions, review before send.

## 9. Open questions and risks

- **Model trust for autonomy.** Flash-tier models on free endpoints are fine for narrated interpretation; are they trustworthy enough for unattended briefing? Mitigation: deterministic computation, cross-check tests, and the M2 review gate.
- **Evaluation.** "Is this briefing good?" needs its own harness — lineage completeness, number cross-checks, and a human rating signal from feedback.
- **Data privacy.** Business data (product, billing, CRM) raises the stakes: local-first execution, secrets management, and per-connector access scoping before production use.
- **Feedback fidelity.** Verbal corrections must be structured before they change definitions; free-text feedback is logged first, applied only when it survives review.
- **Cost and latency.** Scheduled runs on hosted endpoints need budgets and timeouts; the pipeline should degrade to "no briefing, with a reason" rather than a wrong briefing.
