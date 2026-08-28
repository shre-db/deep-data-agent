"""Shared agent streaming: yields structured events for any frontend."""

from typing import Iterator

from langchain_core.messages import AIMessage, ToolMessage

TRACE_TOOL_RESULT_CHARS = 400
TRACE_TOOL_ARGS_CHARS = 200


def _shorten(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tool_message_text(message: ToolMessage) -> str:
    result = message.content
    if isinstance(result, list):  # content blocks
        result = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in result
        )
    return str(result)


def stream_events(agent, user_message: str, thread_id: str) -> Iterator[dict]:
    """Run the agent for one turn, yielding ordered events.

    Event shapes:
      {"type": "text_delta", "text": str}   - streamed agent commentary
      {"type": "tool_call", "name": str, "args": str}
      {"type": "tool_result", "text": str}
      {"type": "final", "text": str}        - the final answer message
    """
    final_content = ""
    for mode, payload in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk = payload[0]
            has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None))
            if isinstance(chunk.content, str) and chunk.content and not has_tool_calls:
                yield {"type": "text_delta", "text": chunk.content}
        elif mode == "updates":
            for update in payload.values():
                for message in (update or {}).get("messages", []):
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for call in message.tool_calls:
                            yield {
                                "type": "tool_call",
                                "name": call["name"],
                                "args": _shorten(call.get("args"), TRACE_TOOL_ARGS_CHARS),
                            }
                    elif isinstance(message, ToolMessage):
                        yield {
                            "type": "tool_result",
                            "text": _shorten(
                                _tool_message_text(message), TRACE_TOOL_RESULT_CHARS
                            ),
                        }
                    if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        final_content = message.content
    if final_content:
        yield {"type": "final", "text": final_content}
