"""CLI entry point for the deep-data-agent prototype."""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.agent import PROJECT_ROOT, build_checkpointer, create_analysis_agent
from app.events import stream_events


def thread_artifacts_dir(thread_id: str) -> Path:
    """Artifacts live in artifacts/<thread_id>/ (thread id path-sanitized)."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", thread_id).strip("_") or "default"
    path = PROJECT_ROOT / "artifacts" / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_agent(agent, user_message: str, thread_id: str) -> str:
    """Run the agent, printing its trace (commands, output, errors) live."""
    final_content = ""
    for event in stream_events(agent, user_message, thread_id):
        kind = event["type"]
        if kind == "commentary":
            print(f"\n{event['text']}\n")
        elif kind == "command":
            print(f"$ {event['text']}")
        elif kind == "output":
            for line in event["text"].splitlines():
                print(f"  {line}")
        elif kind == "error":
            for line in event["text"].splitlines():
                print(f"  ! {line}")
        elif kind == "status":
            mark = "ok" if event["ok"] else "FAILED"
            print(f"  [{mark} exit {event['code']}]")
        elif kind == "tool_call":
            print(f"  [tool call] {event['name']}({event['args']})")
        elif kind == "tool_result":
            print(f"  [tool result] {event['text']}")
        elif kind == "final":
            final_content = event["text"]
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


def ask_question(
    agent, rel_data_path: Path, question: str, thread_id: str, artifacts_dir: Path
) -> None:
    rel_artifacts = artifacts_dir.relative_to(PROJECT_ROOT)
    user_message = (
        f"The dataset is mounted at '{rel_data_path}'.\n"
        f"Save all artifacts (scripts, tables, plots) under '{rel_artifacts}/'.\n"
        f"Answer this question about it:\n\n{question}\n\n"
        "Remember: inspect the dataset, plan, compute with code (never guess "
        "numbers), validate, and end with the required markdown output format."
    )

    print("Running analysis... (live trace below)\n")
    final = run_agent(agent, user_message, thread_id)

    print("\n" + "=" * 60 + "\n")
    print(final or "(no answer produced)")

    artifacts = sorted(p for p in artifacts_dir.iterdir() if p.is_file())
    if artifacts:
        print("\nArtifacts saved:")
        for p in artifacts:
            print(f"- {p.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Natural-language data analysis agent (prototype)."
    )
    parser.add_argument("--data", required=True, help="Path to a CSV file.")
    parser.add_argument(
        "--question", help="Analysis question. If omitted, prompts interactively."
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Conversation thread id. Omit to start a new conversation with "
        "an auto-generated id; pass the same id to resume a previous session.",
    )
    args = parser.parse_args()

    thread_id = args.thread_id or f"session-{datetime.now():%Y%m%d-%H%M%S}"

    data_path = Path(args.data).resolve()
    df = report_dataset(data_path)

    artifacts_dir = thread_artifacts_dir(thread_id)

    rel_data_path = data_path.relative_to(PROJECT_ROOT)
    print("Creating agent...\n")
    agent = create_analysis_agent(checkpointer=build_checkpointer())

    if args.question:
        ask_question(agent, rel_data_path, args.question, thread_id, artifacts_dir)
        return

    # Interactive mode: loop until the user quits, sharing one thread.
    print(
        f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n"
        f"Thread: {thread_id}\n"
        "(conversation and artifacts are remembered per thread; resume later "
        "with --thread-id " + thread_id + ")\n"
        "Type a question, or 'quit' to exit.\n"
    )
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"q", "quit", "exit"}:
            break
        print()
        ask_question(agent, rel_data_path, question, thread_id, artifacts_dir)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
