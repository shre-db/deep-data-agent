"""CLI entry point for the deep-data-agent prototype."""

import argparse
import sys
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from app.agent import PROJECT_ROOT, create_analysis_agent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRACE_TOOL_RESULT_CHARS = 400
TRACE_TOOL_ARGS_CHARS = 200


def _shorten(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_agent(agent, user_message: str) -> str:
    """Run the agent, streaming its trace (tool calls, results, text) live."""
    final_content = ""
    for mode, payload in agent.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            chunk = payload[0]
            has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None))
            if isinstance(chunk.content, str) and chunk.content and not has_tool_calls:
                print(chunk.content, end="", flush=True)
        elif mode == "updates":
            for update in payload.values():
                for message in (update or {}).get("messages", []):
                    if isinstance(message, AIMessage) and message.tool_calls:
                        print()
                        for call in message.tool_calls:
                            args = _shorten(call.get("args"), TRACE_TOOL_ARGS_CHARS)
                            print(f"  [tool call] {call['name']}({args})")
                    elif isinstance(message, ToolMessage):
                        result = message.content
                        if isinstance(result, list):  # content blocks
                            result = " ".join(
                                b.get("text", "") if isinstance(b, dict) else str(b)
                                for b in result
                            )
                        print(f"  [tool result] {_shorten(result, TRACE_TOOL_RESULT_CHARS)}")
                    if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        final_content = message.content
    return final_content


def report_dataset(path: Path) -> pd.DataFrame:
    print(f"Loading dataset...\n")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Error: dataset not found: {path}")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Error: could not read CSV {path}: {e}")
    print(f"Dataset: {path}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}\n")
    print("Columns:")
    for col in df.columns:
        kind = "datetime/string" if col == "date" else str(df[col].dtype)
        print(f"- {col}: {kind}")
    print()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural-language data analysis agent (prototype)."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV file.")
    parser.add_argument(
        "--question", help="Analysis question. If omitted, prompts interactively."
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    df = report_dataset(data_path)

    question = args.question
    if not question:
        print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
        question = input("Question: ").strip()
        if not question:
            sys.exit("Error: no question provided.")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    rel_data_path = data_path.relative_to(PROJECT_ROOT)
    print("Creating agent...\n")
    agent = create_analysis_agent()

    user_message = (
        f"The dataset is mounted at '{rel_data_path}'.\n"
        f"Answer this question about it:\n\n{question}\n\n"
        "Remember: inspect the dataset, plan, compute with code (never guess "
        "numbers), validate, save artifacts under artifacts/, and end with the "
        "required markdown output format."
    )

    print("Running analysis... (live trace below)\n")
    final = run_agent(agent, user_message)

    print("\n" + "=" * 60 + "\n")
    print(final)

    artifacts = sorted(
        p for p in ARTIFACTS_DIR.iterdir()
        if p.is_file() and p.name != ".gitkeep"
    )
    if artifacts:
        print("\nArtifacts saved:")
        for p in artifacts:
            print(f"- artifacts/{p.name}")


if __name__ == "__main__":
    main()
