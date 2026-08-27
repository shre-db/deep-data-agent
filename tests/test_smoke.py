"""Smoke tests: imports, dataset load, agent construction (no LLM calls)."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_imports():
    from app import agent, main, prompts  # noqa: F401


def test_sample_dataset_loads():
    df = pd.read_csv(PROJECT_ROOT / "data" / "sample_sales.csv")
    assert df.shape[0] > 100
    expected_cols = {
        "date", "channel", "product", "region",
        "orders", "revenue", "customers", "conversion_rate",
    }
    assert expected_cols == set(df.columns)


def test_agent_construction():
    from app.agent import create_analysis_agent

    # Construction must not call the LLM; pass a lightweight model spec.
    agent = create_analysis_agent()
    assert agent is not None


def test_inspect_dataset_tool():
    from app.agent import inspect_dataset

    out = inspect_dataset.invoke({"path": "data/sample_sales.csv"})
    assert "rows" in out.lower()
    assert "channel" in out
