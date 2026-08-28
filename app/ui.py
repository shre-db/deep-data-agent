"""Minimal Streamlit frontend for the deep-data-agent.

Run with: uv run streamlit run app/ui.py
"""

import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import PROJECT_ROOT, build_checkpointer, create_analysis_agent
from app.events import classify_messages, stream_events
from app.main import thread_artifacts_dir
from app.markdown_utils import iter_answer_segments, protect_currency


@st.cache_resource
def get_agent():
    return create_analysis_agent(checkpointer=build_checkpointer())


def _new_thread_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"session-{stamp}-{uuid.uuid4().hex[:4]}"


def _model_spec() -> str:
    return dotenv_values(PROJECT_ROOT / ".env").get("MODEL") or "empero:glm-5.3-flash"


def _extract_question(content: str) -> str:
    """Recover the user's raw question from the wrapped prompt message."""
    if "Answer this question about it:" in content:
        tail = content.split("Answer this question about it:", 1)[1]
        question = tail.split("\n\nRemember:", 1)[0].strip()
        if question:
            return question
    return content


def load_thread_history(agent, thread_id: str) -> list[dict] | None:
    """Rebuild chat turns from the checkpointer; None if thread is empty."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = (state.values or {}).get("messages", []) if state.values else []
    if not messages:
        return None

    turns: list[dict] = []
    trace: list[dict] = []
    seen: set = set()
    for m in messages:
        if isinstance(m, HumanMessage):
            turns.append(
                {"role": "user", "content": _extract_question(str(m.content)),
                 "trace": [], "artifacts": []}
            )
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            turns.append(
                {"role": "assistant", "content": m.content,
                 "trace": list(trace), "artifacts": []}
            )
            trace = []
        else:
            trace.extend(classify_messages([m], seen))

    if turns and turns[-1]["role"] == "assistant":
        artifacts_dir = thread_artifacts_dir(thread_id)
        turns[-1]["artifacts"] = sorted(
            p for p in artifacts_dir.iterdir() if p.is_file()
        )
    return turns


def _artifact_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _one_line(text: str, limit: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _path_from_args(args: str) -> str:
    import ast
    import json

    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(args)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict) and "file_path" in parsed:
            return str(parsed["file_path"])
    return ""


def _tool_label(name: str, args: str) -> str:
    if name == "inspect_dataset":
        return "Inspected dataset"
    if name == "ls":
        return "Listed files"
    if name in {"write_file", "edit_file", "read_file"}:
        path = _path_from_args(args)
        verb = {"write_file": "Saved", "edit_file": "Updated", "read_file": "Read"}[name]
        return f"{verb} {path}" if path else name
    return name


class StepCollector:
    """Assemble trace events into renderable steps, incrementally.

    Results attach to their step by tool_call_id (falling back to the most
    recent open step). `add()` returns steps that just completed (ready to
    render); `flush()` returns steps that never received a result. Every
    step is appended to `steps` at creation, so `steps` always holds the
    full ordered list (used by static rendering).
    """

    def __init__(self) -> None:
        self.steps: list[dict] = []
        self._open: dict = {}  # tool_call_id -> step awaiting its result
        self._fallback: dict | None = None  # for id-less events

    def _track(self, step: dict) -> None:
        self.steps.append(step)
        if step.get("id"):
            self._open[step["id"]] = step
        else:
            self._fallback = step

    def _resolve(self, event, pop: bool) -> dict | None:
        sid = event.get("id")
        if sid and sid in self._open:
            step = self._open[sid]
            if pop:
                del self._open[sid]
            return step
        step = self._fallback
        if step is not None and step.get("awaiting_result"):
            if pop:
                self._fallback = None
            return step
        return None

    def add(self, event: dict) -> list[dict]:
        kind = event["type"]
        if kind in {"commentary", "final"}:
            if kind == "commentary":
                self.steps.append({"kind": "note", "text": event["text"]})
            return []
        if kind == "command":
            self._track(
                {"kind": "command", "id": event.get("id"), "command": event["text"],
                 "output": "", "error": "", "status": None, "awaiting_result": True}
            )
            return []
        if kind == "tool_call":
            self._track(
                {"kind": "tool", "id": event.get("id"), "name": event["name"],
                 "args": event["args"], "result": "", "awaiting_result": True}
            )
            return []

        pop = kind in {"status", "tool_result"}
        step = self._resolve(event, pop=pop)
        if step is None:
            if kind in {"output", "error", "tool_result"}:
                # Orphan result: keep it in the step list and surface it now.
                note = {"kind": "note", "text": event["text"], "orphan": True}
                self.steps.append(note)
                return [note]
            return []
        if kind == "output":
            step["output"] = event["text"]
        elif kind == "error":
            step["error"] = event["text"]
        elif kind == "status":
            step["status"] = {"ok": event["ok"], "code": event["code"]}
            step["awaiting_result"] = False
        elif kind == "tool_result":
            step["result"] = event["text"]
            step["awaiting_result"] = False
        return [step] if pop else []

    def flush(self) -> list[dict]:
        leftover = [s for s in self.steps if s.get("awaiting_result")]
        for step in leftover:
            step["awaiting_result"] = False
        self._open.clear()
        self._fallback = None
        return leftover


def group_trace(events: list[dict]) -> list[dict]:
    """Assemble a completed event list into renderable steps."""
    collector = StepCollector()
    for event in events:
        collector.add(event)
    return collector.steps


def render_command_step(step: dict) -> None:
    command = step["command"]
    with st.expander(f"Ran: `{_mono_inline(_one_line(command))}`"):
        st.code(command, language="bash")
        if step["output"]:
            st.code(step["output"])
        # Full stderr (tracebacks included) stays inside the sub-container,
        # whether the command succeeded or failed.
        if step["error"]:
            st.error(_mono_block(protect_currency(step["error"])))
        if step["status"]:
            _mono_caption(f"exit {step['status']['code']} - "
                          f"{'succeeded' if step['status']['ok'] else 'failed'}")


def render_tool_step(step: dict) -> None:
    label = _tool_label(step["name"], step["args"])
    with st.expander(label):
        if step["result"]:
            _mono_caption(step["result"])
        else:
            _mono_caption(f"{step['name']}({_one_line(step['args'], 80)})")


def render_step(step: dict) -> None:
    if step["kind"] == "note":
        if step.get("orphan"):
            with st.expander("Output"):
                st.code(step["text"])
        else:
            _caption(step["text"])
    elif step["kind"] == "command":
        render_command_step(step)
    elif step["kind"] == "tool":
        render_tool_step(step)


def render_steps(steps: list[dict]) -> None:
    for step in steps:
        render_step(step)


def render_trace(events: list[dict]) -> None:
    """Render a completed event list inside the current container."""
    render_steps(group_trace(events))


def _md(text: str) -> None:
    """Render markdown with currency signs protected from LaTeX parsing."""
    st.markdown(protect_currency(text))


def _caption(text: str) -> None:
    """Render a caption with currency signs protected from LaTeX parsing."""
    st.caption(protect_currency(text))


def _mono_inline(text: str) -> str:
    """Escape backticks so `text` can sit inside a markdown code span."""
    return text.replace("`", r"\`")


def _mono_caption(text: str) -> None:
    """Caption rendered as inline code (JetBrains Mono via the code CSS).

    Code-span content is literal, so no currency protection is needed
    (escaping would show a stray backslash).
    """
    st.caption(f"`{_mono_inline(text)}`")


def _mono_block(text: str) -> str:
    """Wrap text in a markdown code fence, growing the fence if the text
    itself contains one."""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}\n{text}\n{fence}"


_MONO_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400..600&display=swap');
code, pre {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
}
</style>
"""


def _inject_fonts() -> None:
    """Load JetBrains Mono and apply it to all code elements.

    Base text is themed to Inter via .streamlit/config.toml; Streamlit has
    no code-font theme option, so code needs this small style block.
    """
    st.html(_MONO_CSS)


def render_answer(content: str, artifacts: list[Path]) -> set[Path]:
    """Render an assistant answer, showing figure references inline.

    Lines of the form `![caption](artifact/path.json)` whose target matches
    an artifact render as charts at that point in the text; everything else
    renders as markdown. Returns the referenced artifacts so the caller can
    exclude them from the trailing gallery.
    """
    referenced: set[Path] = set()
    for kind, payload in iter_answer_segments(content, artifacts):
        if kind == "markdown":
            _md(payload)
        else:
            path = payload["path"]
            referenced.add(path)
            if path.suffix.lower() == ".json":
                _render_plotly_figure(path)
            elif path.suffix.lower() == ".png":
                st.image(str(path))
            elif path.suffix.lower() == ".csv":
                _render_table(path)
            else:
                _caption(_artifact_label(path))
            if payload["caption"]:
                _caption(payload["caption"])
    return referenced


def _render_table(path: Path) -> None:
    """Render a CSV artifact inside a collapsed expander."""
    with st.expander(path.name):
        try:
            st.dataframe(pd.read_csv(path))
        except Exception as e:  # noqa: BLE001
            _mono_caption(f"Could not preview {path.name}: {e}")


def _render_plotly_figure(path: Path) -> None:
    """Render a figure JSON saved by the sandbox chart helpers.

    Re-tints chrome and categorical colors for the active app theme; falls
    back to a plain file caption when the JSON does not parse as a figure.
    """
    import plotly.io as pio

    from app.agent_tools.charts import (
        DARK_CATEGORICAL,
        DARK_CHROME,
        LIGHT_CATEGORICAL,
        LIGHT_CHROME,
    )

    try:
        fig = pio.from_json(path.read_text())
    except Exception:  # noqa: BLE001
        _caption(_artifact_label(path))
        return

    try:
        dark = st.context.theme.get("base") == "dark"
    except Exception:  # noqa: BLE001
        dark = False
    mapping = dict(zip(LIGHT_CATEGORICAL, DARK_CATEGORICAL)) if dark else {}
    chrome = DARK_CHROME if dark else LIGHT_CHROME

    def _tint(value):
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, (list, tuple)):
            return [_tint(v) for v in value]
        return value

    for trace in fig.data:
        for attr in ("marker", "line"):
            obj = getattr(trace, attr, None)
            if obj is not None and getattr(obj, "color", None) is not None:
                obj.color = _tint(obj.color)

    updates = {
        "font_color": chrome["muted"],
        "legend_font_color": chrome["secondary"],
        "xaxis_gridcolor": chrome["gridline"],
        "yaxis_gridcolor": chrome["gridline"],
        "xaxis_zerolinecolor": chrome["baseline"],
        "yaxis_zerolinecolor": chrome["baseline"],
        "xaxis_linecolor": chrome["baseline"],
        "yaxis_linecolor": chrome["baseline"],
    }
    if fig.layout.title and fig.layout.title.text:
        updates["title_font_color"] = chrome["primary"]
    fig.update_layout(**updates)
    # theme=None keeps the figure's own template and palette; the 'streamlit'
    # theme would overwrite the categorical colors baked into the figure.
    st.plotly_chart(fig, theme=None)


def render_artifacts(paths: list[Path]) -> None:
    figures = [p for p in paths if p.suffix.lower() == ".json"]
    pngs = [p for p in paths if p.suffix.lower() == ".png"]
    tables = [p for p in paths if p.suffix.lower() == ".csv"]
    other = [p for p in paths if p not in figures and p not in pngs and p not in tables]
    for p in figures:
        _render_plotly_figure(p)
    for p in pngs:
        st.image(str(p))
    for p in tables:
        _render_table(p)
    for p in other:
        _caption(_artifact_label(p))


def render_assistant_turn(msg: dict) -> None:
    with st.status("Analysis trace", state="complete", expanded=False):
        render_trace(msg.get("trace", []))
    artifacts = msg.get("artifacts", [])
    referenced = render_answer(msg["content"] or "(no answer produced)", artifacts)
    render_artifacts([p for p in artifacts if p not in referenced])


def main() -> None:
    st.set_page_config(page_title="deep-data-agent", page_icon=None, layout="centered")
    _inject_fonts()
    st.title("deep-data-agent")
    st.caption("Ask questions about a CSV dataset; the agent plans, runs pandas "
               "code, and reports evidence-backed results.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = _new_thread_id()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    agent = get_agent()

    with st.sidebar:
        st.subheader("Session")
        st.caption(f"Thread: `{st.session_state.thread_id}`")
        if st.button("New conversation", use_container_width=True):
            st.session_state.thread_id = _new_thread_id()
            st.session_state.messages = []
            st.rerun()
        resume = st.text_input("Resume thread id", placeholder="session-...")
        if st.button("Load thread", disabled=not resume.strip(), use_container_width=True):
            turns = load_thread_history(agent, resume.strip())
            if turns is None:
                st.sidebar.warning(f"No conversation found for thread `{resume.strip()}`.")
            else:
                st.session_state.thread_id = resume.strip()
                st.session_state.messages = turns
                st.rerun()

        st.divider()
        st.subheader("Data")
        data_path = st.text_input(
            "Dataset path", value="data/sample_sales.csv", key="dataset_path"
        )
        st.caption(f"Model: `{_model_spec()}`")
        st.caption("Artifacts are saved under artifacts/<thread>/.")

    artifacts_dir = thread_artifacts_dir(st.session_state.thread_id)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_assistant_turn(msg)
            else:
                _md(msg["content"])

    if not st.session_state.messages:
        st.caption("Start by asking a question about the dataset.")

    question = st.chat_input("Ask a question about the dataset...")
    if not question:
        return

    if not (PROJECT_ROOT / data_path).exists():
        with st.chat_message("assistant"):
            st.error(f"Dataset not found: {data_path}")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        _md(question)

    rel_data_path = (PROJECT_ROOT / data_path).resolve().relative_to(PROJECT_ROOT)
    user_message = (
        f"The dataset is mounted at '{rel_data_path}'.\n"
        f"Save all artifacts (scripts, tables, plots) under "
        f"'{artifacts_dir.relative_to(PROJECT_ROOT)}/'.\n"
        "Note: file-tool results report paths with a leading '/' (virtual "
        "root = the project root). In shell commands use the same path "
        "WITHOUT the leading slash, e.g. 'artifacts/x.py', not '/artifacts/x.py'.\n"
        f"Answer this question about it:\n\n{question}\n\n"
        "Remember: inspect the dataset, plan, compute with code (never guess "
        "numbers), validate, and end with the required markdown output format."
    )

    before = {p for p in artifacts_dir.iterdir() if p.is_file()}

    with st.chat_message("assistant"):
        status = st.status("Analyzing...", state="running", expanded=True)
        trace: list[dict] = []
        final = ""
        # Live rendering renders each step ONCE, complete, via the same
        # render helpers as static history (StepCollector). Results pair to
        # their step by tool_call_id, so multi-call messages keep their own
        # expanders; nothing is written into an expander after creation.
        collector = StepCollector()

        try:
            with status:
                for event in stream_events(agent, user_message, st.session_state.thread_id):
                    trace.append(event)
                    kind = event["type"]
                    if kind == "final":
                        final = event["text"]
                    elif kind == "commentary":
                        _caption(event["text"])
                    else:
                        for step in collector.add(event):
                            render_step(step)
                for step in collector.flush():
                    render_step(step)
            status.update(label="Analysis trace", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()  # server log: full traceback
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(protect_currency(f"Agent error: {e}"))
            # Persist the partial turn so the trace is not lost.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final or "(analysis stopped before completion)",
                    "trace": trace,
                    "artifacts": [],
                }
            )
            return

        after = {p for p in artifacts_dir.iterdir() if p.is_file()}
        new_artifacts = sorted(after - before)
        referenced = render_answer(final or "(no answer produced)", new_artifacts)
        render_artifacts([p for p in new_artifacts if p not in referenced])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final,
            "trace": trace,
            "artifacts": new_artifacts,
        }
    )


main()
