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


def classify_messages(messages) -> Iterator[dict]:
    """Yield trace events from a list of checkpointed LangChain messages.

    Mirrors the event classification of stream_events so restored history
    renders identically to live turns. Agent commentary is recovered from
    tool-call message content.
    """
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            if isinstance(message.content, str) and message.content.strip():
                yield {"type": "commentary", "text": message.content.strip()}
            for call in message.tool_calls:
                if call["name"] == "execute":
                    yield {"type": "command", "text": _execute_command(call.get("args"))}
                else:
                    yield {
                        "type": "tool_call",
                        "name": call["name"],
                        "args": _shorten(call.get("args"), MAX_TOOL_ARGS_CHARS),
                    }
        elif isinstance(message, ToolMessage):
            raw = _tool_message_text(message)
            if message.name == "execute":
                parsed = _parse_execute_output(raw)
                if parsed["stdout"]:
                    yield {"type": "output", "text": _truncate_block(parsed["stdout"])}
                if parsed["stderr"]:
                    yield {"type": "error", "text": _truncate_block(parsed["stderr"])}
                if parsed["status"]:
                    yield {"type": "status", **parsed["status"]}
            else:
                yield {"type": "tool_result", "text": _shorten(raw, MAX_TOOL_RESULT_CHARS)}


def stream_events(agent, user_message: str, thread_id: str) -> Iterator[dict]:
    """Run the agent for one turn, yielding ordered events.

    Event shapes:
      {"type": "commentary", "text": str}    - agent reasoning/commentary block
      {"type": "command", "text": str}       - shell command (execute tool)
      {"type": "output", "text": str}        - command stdout (newlines kept)
      {"type": "error", "text": str}         - command stderr / traceback
      {"type": "status", "ok": bool, "code": int}
      {"type": "tool_call", "name": str, "args": str}   - non-execute tools
      {"type": "tool_result", "text": str}              - non-execute results
      {"type": "final", "text": str}         - the final answer message

    Streamed text is buffered per message id; a buffer whose id matches the
    final answer message is suppressed so the answer is only emitted once
    (as "final"), not duplicated inside the commentary trace.
    """
    final_content = ""
    final_id = None
    buf_id = None
    buf_text = ""

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
                    if isinstance(message, AIMessage) and message.tool_calls:
                        yield from commentary_events()
                        for call in message.tool_calls:
                            if call["name"] == "execute":
                                yield {
                                    "type": "command",
                                    "text": _execute_command(call.get("args")),
                                }
                            else:
                                yield {
                                    "type": "tool_call",
                                    "name": call["name"],
                                    "args": _shorten(call.get("args"), MAX_TOOL_ARGS_CHARS),
                                }
                    elif isinstance(message, ToolMessage):
                        yield from commentary_events()
                        raw = _tool_message_text(message)
                        if message.name == "execute":
                            parsed = _parse_execute_output(raw)
                            if parsed["stdout"]:
                                yield {
                                    "type": "output",
                                    "text": _truncate_block(parsed["stdout"]),
                                }
                            if parsed["stderr"]:
                                yield {
                                    "type": "error",
                                    "text": _truncate_block(parsed["stderr"]),
                                }
                            if parsed["status"]:
                                yield {"type": "status", **parsed["status"]}
                        else:
                            yield {
                                "type": "tool_result",
                                "text": _shorten(raw, MAX_TOOL_RESULT_CHARS),
                            }
                    if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        final_content = message.content
                        final_id = message.id

    if final_content:
        if buf_text.strip() and buf_id != final_id:
            # Trailing commentary from a message other than the final answer.
            yield {"type": "commentary", "text": buf_text.strip()}
        yield {"type": "final", "text": final_content}
