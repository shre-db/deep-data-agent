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

    Streamed text is buffered per message id; a buffer whose id matches the
    final answer message is suppressed so the answer is only emitted once
    (as "final"), not duplicated inside the commentary trace.
    """
    final_content = ""
    final_id = None
    buf_id = None
    buf_text = ""

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
                if buf_id is not None and chunk_id != buf_id and buf_text:
                    # New message started: the previous buffer is commentary.
                    yield {"type": "text_delta", "text": buf_text}
                    buf_text = ""
                buf_id = chunk_id
                buf_text += chunk.content
        elif mode == "updates":
            for update in payload.values():
                for message in (update or {}).get("messages", []):
                    if isinstance(message, AIMessage) and message.tool_calls:
                        if buf_text:
                            yield {"type": "text_delta", "text": buf_text}
                            buf_text = ""
                        for call in message.tool_calls:
                            yield {
                                "type": "tool_call",
                                "name": call["name"],
                                "args": _shorten(call.get("args"), TRACE_TOOL_ARGS_CHARS),
                            }
                    elif isinstance(message, ToolMessage):
                        if buf_text:
                            yield {"type": "text_delta", "text": buf_text}
                            buf_text = ""
                        yield {
                            "type": "tool_result",
                            "text": _shorten(
                                _tool_message_text(message), TRACE_TOOL_RESULT_CHARS
                            ),
                        }
                    if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        final_content = message.content
                        final_id = message.id

    if final_content:
        if buf_text and buf_id != final_id:
            # Trailing commentary from a message other than the final answer.
            yield {"type": "text_delta", "text": buf_text}
        yield {"type": "final", "text": final_content}
