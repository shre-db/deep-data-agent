"""CLI entry point for the deep-data-agent prototype."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from app.agent import PROJECT_ROOT, create_analysis_agent
from app.prompts import ANALYST_PROMPT

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


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

    print("Running analysis...\n")
    result = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    final = result["messages"][-1].content

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
