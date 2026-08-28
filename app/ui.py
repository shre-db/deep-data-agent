"""Minimal Streamlit frontend for the deep-data-agent.

Run with: uv run streamlit run app/ui.py
"""

import uuid
from datetime import datetime
from html import escape
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
            trace.extend(classify_messages([m]))

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


TRACE_CSS = """
<style>
    div[data-testid="stStatusWidget"] .stMarkdown { font-size: 0.85rem; }
    .trace-note {
        color: #6b7280; font-style: italic; background: #f5f6f8;
        border-radius: 10px; padding: 8px 12px; margin: 2px 0;
    }
    .trace-out {
        background: #f5f6f8; border-radius: 10px; padding: 8px 12px;
        margin: 2px 0; font-size: 0.78rem; line-height: 1.35;
        color: #374151; white-space: pre-wrap; word-break: break-word;
        max-height: 260px; overflow-y: auto;
    }
    .trace-err {
        background: #fdecec; border-left: 3px solid #d9534f;
        border-radius: 8px; padding: 8px 12px; margin: 2px 0;
        font-size: 0.78rem; line-height: 1.35; color: #8c2f2b;
        white-space: pre-wrap; word-break: break-word;
        max-height: 260px; overflow-y: auto;
    }
    .trace-ok { color: #2e7d32; font-size: 0.78rem; margin: 2px 0 8px 0; }
    .trace-fail { color: #c62828; font-size: 0.78rem; margin: 2px 0 8px 0; }
    .trace-tool { font-size: 0.85rem; margin: 6px 0 2px 0; }
</style>
"""


def _styled(css_class: str, text: str) -> None:
    st.markdown(
        f'<div class="{css_class}">{escape(text)}</div>', unsafe_allow_html=True
    )


def render_event(event: dict) -> None:
    """Render one trace event as a visually distinct block.

    Used by both the live streaming loop and static history rendering so
    the two always look identical.
    """
    kind = event["type"]
    if kind == "commentary":
        _styled("trace-note", event["text"])
    elif kind == "command":
        st.code(event["text"], language="bash")
    elif kind == "output":
        _styled("trace-out", event["text"])
    elif kind == "error":
        _styled("trace-err", event["text"])
    elif kind == "status":
        word = "succeeded" if event["ok"] else "failed"
        css = "trace-ok" if event["ok"] else "trace-fail"
        _styled(css, f"exit {event['code']} - {word}")
    elif kind == "tool_call":
        st.markdown(
            f'<div class="trace-tool"><strong>{escape(event["name"])}</strong> '
            f'<code>{escape(event["args"])}</code></div>',
            unsafe_allow_html=True,
        )
    elif kind == "tool_result":
        _styled("trace-note", event["text"])
    elif kind == "final":
        return


def render_trace(events: list[dict]) -> None:
    """Render a completed event list inside the current container."""
    for event in events:
        render_event(event)


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
    st.markdown(TRACE_CSS, unsafe_allow_html=True)
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

        try:
            with status:
                for event in stream_events(agent, user_message, st.session_state.thread_id):
                    trace.append(event)
                    if event["type"] == "final":
                        final = event["text"]
                    else:
                        render_event(event)
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
