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
                note = {"kind": "note", "text": event["text"]}
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
    failed = bool(step["status"] and not step["status"]["ok"])
    with st.expander(f"Ran: {_one_line(command)}"):
        st.code(command, language="bash")
        if step["output"]:
            st.code(step["output"])
        if step["error"] and not failed:
            st.error(step["error"])
        if step["status"]:
            st.caption(f"exit {step['status']['code']} - "
                       f"{'succeeded' if step['status']['ok'] else 'failed'}")
    if failed:
        if step["error"]:
            st.error(step["error"])
        if step["status"]:
            st.error(f"Command failed with exit code {step['status']['code']}")


def render_tool_step(step: dict) -> None:
    label = _tool_label(step["name"], step["args"])
    with st.expander(label):
        if step["result"]:
            st.caption(step["result"])
        else:
            st.caption(f"{step['name']}({_one_line(step['args'], 80)})")


def render_step(step: dict) -> None:
    if step["kind"] == "note":
        st.caption(step["text"])
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


def render_artifacts(paths: list[Path]) -> None:
    pngs = [p for p in paths if p.suffix.lower() == ".png"]
    tables = [p for p in paths if p.suffix.lower() == ".csv"]
    other = [p for p in paths if p not in pngs and p not in tables]
    for p in pngs:
        st.image(str(p))
    for p in tables:
        with st.expander(p.name):
            try:
                st.dataframe(pd.read_csv(p))
            except Exception as e:  # noqa: BLE001
                st.caption(f"Could not preview {p.name}: {e}")
    for p in other:
        st.caption(_artifact_label(p))


def render_assistant_turn(msg: dict) -> None:
    with st.status("Analysis trace", state="complete", expanded=False):
        render_trace(msg.get("trace", []))
    st.markdown(msg["content"] or "(no answer produced)")
    render_artifacts(msg.get("artifacts", []))


def main() -> None:
    st.set_page_config(page_title="deep-data-agent", page_icon=None, layout="centered")
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
                st.markdown(msg["content"])

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
        st.markdown(question)

    rel_data_path = (PROJECT_ROOT / data_path).resolve().relative_to(PROJECT_ROOT)
    user_message = (
        f"The dataset is mounted at '{rel_data_path}'.\n"
        f"Save all artifacts (scripts, tables, plots) under "
        f"'{artifacts_dir.relative_to(PROJECT_ROOT)}/'.\n"
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
                        st.caption(event["text"])
                    else:
                        for step in collector.add(event):
                            render_step(step)
                for step in collector.flush():
                    render_step(step)
            status.update(label="Analysis trace", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(f"Agent error: {e}")
            return

        st.markdown(final or "(no answer produced)")

        after = {p for p in artifacts_dir.iterdir() if p.is_file()}
        new_artifacts = sorted(after - before)
        render_artifacts(new_artifacts)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final,
            "trace": trace,
            "artifacts": new_artifacts,
        }
    )


main()
