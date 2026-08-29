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

## 3. Scope and domain generality

The core loop is domain-invariant: data → analyst core → validated run →
deliverable → audience → feedback → memory is the shape of all knowledge
work, from toy CSV questions to astrophysics. Whether the **system** works
in practice for a given domain is a separate question, decided by variables
that surround the loop rather than by its shape:

- **Data substrate** — from one CSV to petabyte archives, domain formats,
  and calibration pipelines. The data node hides an entire infrastructure
  budget.
- **Who authors the analysis** — the agent writes `analysis.py` for a
  business dataset; in frontier domains the analysis is community code
  developed over years, and the agent's job shifts from authoring to
  orchestrating and checking established codes.
- **The nature of validation** — cross-checks and lineage suffice where
  truth is computable from the data; where truth is provisional (theory,
  simulations, other instruments), validation is a discipline of its own.
- **Audience** — the analyst-on-staff pitch is translating expert work for
  non-experts; where the audience is the expert, the value shifts to
  coverage, speed, and error-catching.
- **Memory** — business definitions and baselines in this domain; in
  science, memory grows into literature: prior art, citations, what has
  been falsified.
- **Feedback thickness** — verbal corrections work for definitions; peer
  review is structured, adversarial, and years-delayed.
- **Consequence weight** — a wrong briefing annoys; a wrong result can
  misdirect instruments or policy. The action gates must be re-weighted
  accordingly.

The claim here is narrower than the loop: this design is for
data-to-decision work where a human owns the judgment and the data fits the
team's own machines. Extending it to frontier domains does not change the
loop — it upgrades each node into a subsystem.

### Worked examples

**Serves as-is** — domains where the surrounding variables sit inside the
architecture's budget (the standard analyst-on-staff architecture, M1–M4,
not today's CSV-only prototype):

| Domain | Typical question | Why it fits |
|---|---|---|
| SaaS / e-commerce analytics | "Why did activation drop last week?" | The canonical case: warehouse/CSV-scale data, agent-authored SQL+pandas, cross-checkable numbers, exec↔analyst translation, quick verbal feedback, contained consequences. |
| Marketing performance | "Which campaigns drove this quarter's pipeline?" | Small structured data; standard funnels and ROAS are computable and checkable; audience altitude (CMO vs analyst) is the core feature. |
| Operations & support analytics | "Why is ticket resolution time rising?" | Trends and segments over a small dataset; decisions stay with a human owner; definitions fit the memory store perfectly. |
| SMB finance / monthly close | "Explain the variance vs budget." | Numbers must tie out — exactly what cross-checks and lineage are for; the definitions store carries the accounting conventions. |
| Team-level sports analytics | "Is player load driving injury risk?" | Small stats datasets; metrics are clear; coach vs analyst rendering is the same altitude feature as VP vs analyst. |
| Small-city civic data | "Where is building-permit volume growing?" | Public, low-volume data; standard aggregations; higher stakes than SaaS, but the approval-gated action layer covers it. |

**Too taxing for the standard architecture** — each row names which
surrounding variables break it; the loop still holds, the system doesn't:

| Domain | Typical question | What breaks |
|---|---|---|
| Astrophysics / particle physics | "Is this excess significant?" | **Data substrate** (petabytes, FITS/ROOT, calibration pipelines), **authorship** (community codes built over decades — the agent orchestrates, doesn't write `analysis.py`), **validation** (truth is provisional: theory, simulations, systematics), **memory** (literature, not definitions), **feedback** (peer review, years-delayed), **audience** (experts — no translation value). Every variable at max tax. |
| Healthcare / clinical analytics | "Does this intervention reduce readmissions?" | **Consequence weight** (patient harm, liability — gates become certification), **validation** (statistical cross-checks don't establish clinical validity; coding artifacts and missingness are informative and need domain judgment), **feedback** (regulatory, slow, adversarial). Data can be small; that's not the problem. |
| Trading / market analytics | "Is this strategy alpha?" | **Data substrate** (high-frequency, non-stationary) and an **adversarial environment** — the market moves against you. Validation needs out-of-sample discipline beyond lineage; latency kills the human-gated action loop. |
| Climate science | "What does this ensemble say about regional rainfall?" | **Data substrate** (model outputs, ensembles, netCDF), **authorship and validation** (community-standard workflows and intercomparisons, not authorable per run). Partial transfer: the briefing-and-audience layer fits policymakers — a good "parts transfer, parts don't" case. |
| Cybersecurity / threat detection | "Is this endpoint compromised?" | **Adversarial + latency**: attackers adapt; detection is a real-time system, not a morning briefing. The human-approval action gate is too slow; false-positive costs are immediate. |
| Legal / e-discovery | "Does this contract expose us?" | **Validation** (truth is authority-based — precedent, not cross-checkable numbers) and **memory** (the entire corpus of case law is context). Textual, argumentative truth is outside the loop's compute-and-verify assumption. |

## 4. Design principles

1. **Deterministic plumbing, LLM brains.** Scheduling, connectors, storage, and delivery are plain code. The model interprets and narrates; it never sits in the correctness-critical path of computing a number.
2. **Analyses as code.** Every repeatable analysis is a versioned, vetted script. Edits are reviewed; re-runs diff against history. A churn number is computed by an approved pipeline, not regenerated from scratch each morning.
3. **Lineage everywhere.** Every number in a deliverable cites its dataset, script version, and run. No orphan statistics.
4. **Audience is a rendering policy.** Expertise level is a property of the deliverable, not something inferred from conversation. One analysis, N renderings.
5. **Feedback is memory.** Verbal reinforcement learning in the pragmatic sense: corrections and ratings become definition updates and few-shot context for future runs — inspectable, no fine-tuning.
6. **Actions are gated.** Creating tickets and sending alerts start behind human approval. The trust bar for acting on someone's behalf is higher than the bar for informing them.

## 5. Architecture

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

## 6. Data model

| Entity | Key fields | Notes |
|---|---|---|
| `business_definition` | name, definition, owner, version, supersedes | "When I say churn, I mean X." Versioned; corrections append, never silently overwrite |
| `audience_profile` | person, role, expertise, channel, preferences | Asked once per recipient, stored |
| `run` | id, trigger (schedule/manual), status, lineage refs | A thread + its metadata |
| `artifact` | path, kind, source_data_hash, script_version, run_id | The lineage sidecar |
| `briefing` | run_id, audience, rendered content, delivery status | One run renders N briefings |
| `feedback` | briefing_id, kind (correction/rating), text | Corrections may spawn definition updates |
| `task` | created_from, assignee, severity, status | The action layer's output |

## 7. Canonical scenario — the morning briefing

1. **07:00** — the scheduler starts the daily run: product, billing, and CRM data, each freshness-checked.
2. **Analyst** — the core runs the vetted pipelines for activation, retention, and revenue; diffs against historical baselines.
3. **Anomaly** — retention in the SMB segment shows an accelerating decline. The deterministic baseline flags it; the model attributes likely causes — a billing API error spike and a pricing change on Tuesday — confirmed against external sources via web search.
4. **Validation** — cross-check tests pass; lineage is attached to every cited number, including web-sourced facts (each with its URL).
5. **Rendering** — the briefing renders at two altitudes: a five-minute narrative for the VP Product (one chart per finding, decisions needed) and a full version for the analyst (methodology, caveats, scripts).
6. **Delivery** — the VP's briefing lands in Slack with a review gate; the analyst's version lands in their inbox.
7. **Escalation** — churn severity crosses threshold: the system proposes an investigation task (assignee, severity, lineage links) for human approval.
8. **Feedback** — the VP replies "by churn I meant paid-only accounts." The correction becomes a new version of the `churn` definition, and tomorrow's run uses it.

## 8. Roadmap

| Milestone | Scope | Success criteria |
|---|---|---|
| **M0 — Chat prototype** | Today's app | (done) |
| **M1 — Briefing + knowledge** | Briefing run type, definitions store, audience profiles | One daily briefing renders at two altitudes from stored definitions |
| **M2 — Schedule + delivery** | Scheduler, Slack/email renderer, human review gate | Briefings arrive without opening the app; console used for review |
| **M3 — Baselines + external context** | Deterministic stats (seasonality, z-scores) feeding findings; a narrow web-search tool (Parallel Web Systems) for attribution | Anomalies are cited against historical patterns and external events, with confidence |
| **M4 — Action layer** | Investigation tasks, severity routing, approval flows | Escalation creates a tracked task with a lineage link |

**Web search (M3 scope).** When the agent needs external context — known
outages, pricing changes, industry benchmarks — it uses a narrow search
tool backed by **Parallel Web Systems**. Rules: search only for external
context, never to compute from the dataset; every sourced fact must cite
its URL and renders separately from computed numbers; enabled by env
config, so the agent degrades gracefully without it.

## 9. Mapping to today's codebase

- `app/agent.py` — Deep Agent + `LocalShellBackend` + the `analysis.py` overwrite pattern → the analyst core and the seed of analyses-as-code.
- `app/events.py` — the structured event trace → the run record briefings render from.
- `app/agent_tools/charts.py` — the standardized chart language → the visual layer of briefings.
- `app/prompts.py` — the answer framework → the analyst's job description; audience profile and definitions become injected context.
- Checkpointer + `artifacts/<thread>/` → the run store; lineage arrives as metadata sidecars per artifact.
- `app/ui.py` — the console: inspect runs, correct definitions, review before send.

## 10. Open questions and risks

- **Model trust for autonomy.** Flash-tier models on free endpoints are fine for narrated interpretation; are they trustworthy enough for unattended briefing? Mitigation: deterministic computation, cross-check tests, and the M2 review gate.
- **Evaluation.** "Is this briefing good?" needs its own harness — lineage completeness, number cross-checks, and a human rating signal from feedback.
- **Data privacy.** Business data (product, billing, CRM) raises the stakes: local-first execution, secrets management, and per-connector access scoping before production use.
- **Feedback fidelity.** Verbal corrections must be structured before they change definitions; free-text feedback is logged first, applied only when it survives review.
- **Cost and latency.** Scheduled runs on hosted endpoints need budgets and timeouts; the pipeline should degrade to "no briefing, with a reason" rather than a wrong briefing.
