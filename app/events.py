"""Shared agent streaming: yields structured events for any frontend."""

import json
import re
from typing import Iterator

from langchain_core.messages import AIMessage, ToolMessage

# Output block limits (lines preserved; only whole lines are dropped)
MAX_OUTPUT_LINES_HEAD = 30
MAX_OUTPUT_LINES_TAIL = 10
MAX_OUTPUT_CHARS = 4000
MAX_TOOL_RESULT_CHARS = 400
MAX_TOOL_ARGS_CHARS = 200

_STATUS_RE = re.compile(r"^\[Command (succeeded|failed) with exit code (\d+)\]$")
_EXIT_CODE_RE = re.compile(r"^Exit code: (\d+)$")


def _shorten(text, limit: int) -> str:
    """Collapse whitespace for single-line uses (tool args, short results)."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_message_text(message: ToolMessage) -> str:
    result = message.content
    if isinstance(result, list):  # content blocks
        result = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in result
        )
    return str(result)


def _truncate_block(text: str) -> str:
    """Cap a multi-line block by lines and chars, keeping head and tail."""
    lines = text.splitlines()
    total = len(lines)
    if total > MAX_OUTPUT_LINES_HEAD + MAX_OUTPUT_LINES_TAIL + 1:
        head = lines[:MAX_OUTPUT_LINES_HEAD]
        tail = lines[-MAX_OUTPUT_LINES_TAIL:]
        hidden = total - MAX_OUTPUT_LINES_HEAD - MAX_OUTPUT_LINES_TAIL
        lines = [*head, f"... [{hidden} lines hidden] ...", *tail]
    out = "\n".join(lines)
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
    return out


def _parse_execute_output(raw: str) -> dict:
    """Split deepagents execute output into stdout / stderr / status.

    Raw format (see deepagents.middleware.filesystem._format_execute_output
    and backends/local_shell.py):
        stdout lines
        [stderr] line            (one per stderr line)
        Exit code: N             (backend, non-zero exits)
        [Command succeeded|failed with exit code N]
        [Output was truncated due to size limits]   (optional)
    """
    stdout: list[str] = []
    stderr: list[str] = []
    status: dict | None = None

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("[stderr] "):
            stderr.append(stripped[len("[stderr] "):])
            continue
        m = _STATUS_RE.match(stripped)
        if m:
            status = {"ok": m.group(1) == "succeeded", "code": int(m.group(2))}
            continue
        m = _EXIT_CODE_RE.match(stripped)
        if m and status is None:
            code = int(m.group(1))
            status = {"ok": code == 0, "code": code}
            continue
        if stripped in {
            "[Output was truncated due to size limits]",
            "[Output exceeded the capture size limit and was truncated; "
            "the saved file is incomplete]",
        }:
            continue
        stdout.append(line)

    text = "\n".join(stdout).strip("\n")
    if text == "<no output>":
        text = ""
    return {"stdout": text, "stderr": "\n".join(stderr), "status": status}


def _execute_command(args) -> str:
    """Extract the shell command string from execute tool args."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    if isinstance(args, dict):
        return str(args.get("command", ""))
    return str(args)


def _message_events(
    message, seen: set, include_commentary: bool, tool_names: dict
) -> Iterator[dict]:
    """Yield trace events for one LangChain message.

    Messages whose id was already seen are skipped (LangGraph node updates
    can re-emit the same message via middleware nodes). Messages without an
    id are always processed.

    `tool_names` maps tool_call_id -> tool name, built from AIMessage
    tool_calls; it is the reliable way to classify ToolMessages, whose
    `name` attribute may be missing on streamed objects.
    """
    mid = getattr(message, "id", None)
    if mid is not None:
        if mid in seen:
            return
        seen.add(mid)

    if isinstance(message, AIMessage) and message.tool_calls:
        if include_commentary and isinstance(message.content, str) and message.content.strip():
            yield {"type": "commentary", "text": message.content.strip()}
        for call in message.tool_calls:
            tool_names[call["id"]] = call["name"]
            if call["name"] == "execute":
                yield {"type": "command", "text": _execute_command(call.get("args")), "id": call["id"]}
            else:
                yield {
                    "type": "tool_call",
                    "name": call["name"],
                    "args": _shorten(call.get("args"), MAX_TOOL_ARGS_CHARS),
                    "id": call["id"],
                }
    elif isinstance(message, ToolMessage):
        raw = _tool_message_text(message)
        tcid = getattr(message, "tool_call_id", None)
        name = message.name or tool_names.get(tcid)
        if name == "execute":
            parsed = _parse_execute_output(raw)
            for event in (
                {"type": "output", "text": _truncate_block(parsed["stdout"])},
                {"type": "error", "text": _truncate_block(parsed["stderr"])},
            ):
                if event["text"]:
                    yield {**event, "id": tcid}
            if parsed["status"]:
                yield {**{"type": "status", **parsed["status"]}, "id": tcid}
        else:
            yield {
                "type": "tool_result",
                "text": _shorten(raw, MAX_TOOL_RESULT_CHARS),
                "id": tcid,
            }
    elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
        yield {"type": "final", "text": message.content}


def classify_messages(messages, seen: set | None = None) -> Iterator[dict]:
    """Yield trace events from a list of checkpointed LangChain messages.

    Mirrors the event classification of stream_events so restored history
    renders identically to live turns. Agent commentary is recovered from
    tool-call message content.
    """
    seen = set() if seen is None else seen
    tool_names: dict = {}
    for message in messages:
        yield from _message_events(message, seen, include_commentary=True, tool_names=tool_names)


def stream_events(agent, user_message: str, thread_id: str) -> Iterator[dict]:
    """Run the agent for one turn, yielding ordered events.

    Event shapes (step events carry `id` = tool_call_id for pairing):
      {"type": "commentary", "text": str}    - agent reasoning/commentary block
      {"type": "command", "text": str, "id": str}       - shell command
      {"type": "output", "text": str, "id": str}        - command stdout
      {"type": "error", "text": str, "id": str}         - command stderr
      {"type": "status", "ok": bool, "code": int, "id": str}
      {"type": "tool_call", "name": str, "args": str, "id": str}
      {"type": "tool_result", "text": str, "id": str}
      {"type": "final", "text": str}         - the final answer message

    Streamed text is buffered per message id; a buffer whose id matches the
    final answer message is suppressed so the answer is only emitted once
    (as "final"), not duplicated inside the commentary trace.
    """
    final_content = ""
    final_id = None
    buf_id = None
    buf_text = ""
    seen: set = set()
    tool_names: dict = {}

    def commentary_events() -> Iterator[dict]:
        nonlocal buf_id, buf_text
        if buf_text.strip():
            yield {"type": "commentary", "text": buf_text.strip()}
        buf_id, buf_text = None, ""

    for mode, payload in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk = payload[0]
            has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None))
            if isinstance(chunk.content, str) and chunk.content and not has_tool_calls:
                chunk_id = getattr(chunk, "id", None)
                if buf_id is not None and chunk_id != buf_id and buf_text.strip():
                    # New message started: the previous buffer is commentary.
                    yield {"type": "commentary", "text": buf_text.strip()}
                    buf_text = ""
                buf_id = chunk_id
                buf_text += chunk.content
        elif mode == "updates":
            for update in payload.values():
                for message in (update or {}).get("messages", []):
                    for event in _message_events(
                        message, seen, include_commentary=False, tool_names=tool_names
                    ):
                        if event["type"] == "final":
                            # Record without flushing the stream buffer: the
                            # buffered text with this id IS the final answer
                            # and is suppressed via the final_id check below.
                            final_content = event["text"]
                            final_id = getattr(message, "id", None)
                            continue
                        yield from commentary_events()
                        yield event

    if final_content:
        if buf_text.strip() and buf_id != final_id:
            # Trailing commentary from a message other than the final answer.
            yield {"type": "commentary", "text": buf_text.strip()}
        yield {"type": "final", "text": final_content}
