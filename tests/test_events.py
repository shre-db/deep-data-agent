"""Offline tests for the shared event generator (no LLM calls)."""

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.events import stream_events


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
            {"tools": {"messages": [ToolMessage(content="shape info", tool_call_id="1")]}},
        )
        # Final answer message: streams with id m2, then completes with id m2.
        yield ("messages", (AIMessageChunk(content="Final ", id="m2"), {}))
        yield ("messages", (AIMessageChunk(content="answer", id="m2"), {}))
        yield ("updates", {"model": {"messages": [AIMessage(content="Final answer", id="m2")]}})


def test_stream_events_order_and_shapes():
    events = list(stream_events(StubAgent(), "question", "thread-1"))
    assert [e["type"] for e in events] == [
        "text_delta",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[0]["text"] == "Hello "
    assert events[1]["name"] == "inspect_dataset"
    assert events[1]["args"] == "{'path': 'data/x.csv'}"
    assert events[2]["text"] == "shape info"
    assert events[3]["text"] == "Final answer"


def test_final_answer_not_duplicated_in_trace():
    events = list(stream_events(StubAgent(), "question", "thread-1"))
    commentary = "".join(e["text"] for e in events if e["type"] == "text_delta")
    assert "Final answer" not in commentary
    assert "Hello" in commentary


def test_stream_events_truncates_tool_results():
    class LongAgent(StubAgent):
        def stream(self, _input, config=None, stream_mode=None):
            yield (
                "updates",
                {"tools": {"messages": [ToolMessage(content="x" * 1000, tool_call_id="1")]}},
            )

    events = list(stream_events(LongAgent(), "q", "t"))
    assert len(events[0]["text"]) < 500
    assert events[0]["text"].endswith("...")
