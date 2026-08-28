"""Construction of the Deep Agents analysis agent."""

import os
import sys
from pathlib import Path

import pandas as pd
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv
from langchain_core.tools import tool

from app.prompts import ANALYST_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "empero": {
        "base_url": "https://free.empero.org/v1",
        "key_env": "EMPERO_API_KEY",
        "key_required": False,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "key_required": True,
    },
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "key_required": True,
    },
}


def build_model(model_spec: str | None = None):
    """Build a chat model from a MODEL spec like 'empero:glm-5.3-flash'."""
    load_dotenv(PROJECT_ROOT / ".env")
    model_spec = model_spec or os.environ.get("MODEL") or "empero:glm-5.3-flash"
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        raise ValueError(
            f"MODEL must look like 'provider:model' (e.g. 'empero:glm-5.3-flash'), got: {model_spec!r}"
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown model provider {provider!r}. Supported: {sorted(PROVIDERS)}"
        )
    cfg = PROVIDERS[provider]
    api_key = os.environ.get(cfg["key_env"]) or "not-needed"
    if cfg["key_required"] and api_key == "not-needed":
        raise ValueError(
            f"Model {model_spec!r} requires an API key. "
            f"Set {cfg['key_env']} in your .env file."
        )
    from langchain_openai import ChatOpenAI

    kwargs = {"model": model_name, "api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    # Cap max_tokens so constrained/free provider tiers with small credit
    # balances don't reject the request outright (override with MAX_TOKENS).
    try:
        kwargs["max_tokens"] = int(os.environ.get("MAX_TOKENS", "8192"))
    except ValueError:
        pass
    return ChatOpenAI(**kwargs)


@tool
def inspect_dataset(path: str) -> str:
    """Inspect a CSV dataset: shape, columns, dtypes, missing values, and a preview.

    Args:
        path: Path to the CSV file, relative to the project root.
    """
    try:
        df = pd.read_csv(PROJECT_ROOT / path)
    except Exception as e:  # noqa: BLE001
        return f"Error reading {path!r}: {e}"
    buf = [
        f"File: {path}",
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "\nColumns:",
    ]
    for col in df.columns:
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        buf.append(
            f"- {col}: dtype={df[col].dtype}, missing={missing}, unique={unique}, "
            f"sample={df[col].dropna().head(3).tolist()}"
        )
    buf.append("\nFirst 5 rows:")
    buf.append(df.head(5).to_string())
    buf.append("\nNumeric summary:")
    buf.append(df.describe(include="number").to_string())
    return "\n".join(buf)


def _build_backend() -> LocalShellBackend:
    venv_bin = Path(sys.executable).parent
    env = {
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": "/tmp",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": "/tmp/mplconfig",
        "LANG": "C.UTF-8",
    }
    Path("/tmp/mplconfig").mkdir(exist_ok=True)
    return LocalShellBackend(
        root_dir=str(PROJECT_ROOT),
        virtual_mode=True,
        inherit_env=False,
        env=env,
        timeout=180,
    )


def create_analysis_agent(model=None, checkpointer=None):
    """Create the data analysis Deep Agent.

    Kept as a factory so specialized subagents (analyst/reviewer) can be
    added later via the `subagents` parameter.

    Args:
        model: Optional pre-built chat model; defaults to build_model().
        checkpointer: Optional LangGraph checkpointer for multi-turn memory.
    """
    return create_deep_agent(
        model=model or build_model(),
        tools=[inspect_dataset],
        system_prompt=ANALYST_PROMPT,
        backend=_build_backend(),
        checkpointer=checkpointer,
    )


def build_checkpointer(db_path: Path | None = None):
    """Create a SQLite checkpointer persisted under .checkpoints/."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = db_path or PROJECT_ROOT / ".checkpoints" / "threads.db"
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)
