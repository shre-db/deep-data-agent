"""Offline tests for the shared event generator (no LLM calls)."""

import json

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.events import _parse_execute_output, _truncate_block, stream_events


class StubAgent:
    """Mimics the LangGraph stream surface used by stream_events."""

    def stream(self, _input, config=None, stream_mode=None):
        assert stream_mode == ["messages", "updates"]
        yield ("messages", (AIMessageChunk(content="Hello ", id="m1", tool_call_chunks=[]), {}))
        yield (
            "updates",
            {
                "model": {
                    "messages": [
                        AIMessage(
                            content="",
                            id="m1",
                            tool_calls=[
                                {
                                    "name": "inspect_dataset",
                                    "args": {"path": "data/x.csv"},
                                    "id": "1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                },
                "tools": None,  # LangGraph can emit None updates
            },
        )
        yield (
            "updates",
            {"tools": {"messages": [ToolMessage(content="shape info", tool_call_id="1", name="inspect_dataset")]}},
        )
        # An execute turn: command args + deepagents-formatted output
        yield (
            "updates",
            {
                "model": {
                    "messages": [
                        AIMessage(
                            content="",
                            id="m1",
                            tool_calls=[
                                {
                                    "name": "execute",
                                    "args": {"command": "python analysis.py"},
                                    "id": "2",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }
            },
        )
        yield (
            "updates",
            {
                "tools": {
                    "messages": [
                        ToolMessage(
                            content=(
                                "row 1\nrow 2\n"
                                "[stderr] warning: something\n"
                                "\n[Command failed with exit code 3]"
                            ),
                            tool_call_id="2",
                            name="execute",
                        )
                    ]
                }
            },
        )
        # Final answer message: streams with id m2, then completes with id m2.
        yield ("messages", (AIMessageChunk(content="Final ", id="m2"), {}))
        yield ("messages", (AIMessageChunk(content="answer", id="m2"), {}))
        yield ("updates", {"model": {"messages": [AIMessage(content="Final answer", id="m2")]}})


def test_stream_events_order_and_shapes():
    events = list(stream_events(StubAgent(), "question", "thread-1"))
    assert [e["type"] for e in events] == [
        "commentary",
        "tool_call",
        "tool_result",
        "command",
        "output",
        "error",
        "status",
        "final",
    ]
    assert events[0]["text"] == "Hello"
    assert events[1]["name"] == "inspect_dataset"
    assert events[2]["text"] == "shape info"
    assert events[3]["text"] == "python analysis.py"
    assert events[4]["text"] == "row 1\nrow 2"  # newlines preserved
    assert "warning: something" in events[5]["text"]
    assert events[6] == {"type": "status", "ok": False, "code": 3}
    assert events[7]["text"] == "Final answer"


def test_final_answer_not_duplicated_in_trace():
    events = list(stream_events(StubAgent(), "question", "thread-1"))
    commentary = "".join(e["text"] for e in events if e["type"] == "commentary")
    assert "Final answer" not in commentary
    assert "Hello" in commentary


def test_parse_execute_output_success():
    parsed = _parse_execute_output("hello\nworld\n\n[Command succeeded with exit code 0]")
    assert parsed["stdout"] == "hello\nworld"
    assert parsed["stderr"] == ""
    assert parsed["status"] == {"ok": True, "code": 0}


def test_parse_execute_output_stderr_and_exit_code_line():
    parsed = _parse_execute_output(
        "out\n[stderr] boom\nExit code: 2\n\n[Command failed with exit code 2]"
    )
    assert parsed["stdout"] == "out"
    assert parsed["stderr"] == "boom"
    assert parsed["status"] == {"ok": False, "code": 2}


def test_parse_execute_output_no_output_marker():
    parsed = _parse_execute_output("<no output>\n\n[Command succeeded with exit code 0]")
    assert parsed["stdout"] == ""


def test_truncate_block_keeps_head_and_tail():
    text = "\n".join(f"line {i}" for i in range(100))
    out = _truncate_block(text)
    lines = out.splitlines()
    assert lines[0] == "line 0"
    assert lines[-1] == "line 99"
    assert "lines hidden" in out
    assert len(lines) == 30 + 1 + 10


def test_execute_command_extraction():
    from app.events import _execute_command

    assert _execute_command({"command": "ls -la"}) == "ls -la"
    assert _execute_command(json.dumps({"command": "echo hi"})) == "echo hi"
    assert _execute_command("raw string") == "raw string"


def test_long_tool_result_shortened():
    class LongAgent(StubAgent):
        def stream(self, _input, config=None, stream_mode=None):
            yield (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content="x" * 1000, tool_call_id="1", name="inspect_dataset"
                            )
                        ]
                    }
                },
            )

    events = list(stream_events(LongAgent(), "q", "t"))
    assert len(events[0]["text"]) < 500
    assert events[0]["text"].endswith("...")
