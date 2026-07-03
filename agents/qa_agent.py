"""The Q&A agent — answers follow-up questions using LLM-generated pandas code."""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from utils.data_loader import Dataset
from utils.safe_exec import safe_execute, ExecutionResult
from prompts.templates import build_qa_messages, build_explain_messages

load_dotenv()

# Max turns of history we send to the LLM. Prevents context bloat.
MAX_HISTORY_TURNS = 4


@dataclass
class QATurn:
    """One question/answer exchange."""
    question: str
    code: str
    success: bool
    value: Any = None
    figure: Any = None
    error: str | None = None
    stdout: str = ""
    explanation: str = ""


def _get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1,
        streaming=False,
        max_tokens=1024,
    )


def _format_schema(datasets: dict[str, Dataset]) -> str:
    """Compact schema string for one or more datasets. Each dataset is
    presented with its variable name (e.g. `df_sales`), source file,
    shape, and per-column type + samples. Sample values are critical:
    they're often what determines whether a filter should be `== 'US'`
    or `== 'United States'`."""
    lines = []
    for var_name, dataset in datasets.items():
        n_rows = len(dataset.df)
        n_cols = dataset.df.shape[1]
        lines.append(
            f"\n▸ `{var_name}` ({n_rows:,} rows × {n_cols} cols, "
            f"source: {dataset.source_name})"
        )
        for p in dataset.profiles:
            samples = ", ".join(str(v) for v in p.sample_values[:3])
            lines.append(
                f"    - `{p.name}` ({p.kind}, dtype={p.dtype}) — samples: {samples}"
            )
    return "\n".join(lines).strip()


def _format_history(history: list[QATurn], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Include only the last N turns, summarized. Keeps LLM context small."""
    if not history:
        return "(none yet)"
    recent = history[-max_turns:]
    parts = []
    for i, turn in enumerate(recent, start=1):
        parts.append(f"Turn {i}: user asked {turn.question!r}")
        if turn.success:
            parts.append(f"  → code executed successfully")
        else:
            parts.append(f"  → failed: {turn.error}")
    return "\n".join(parts)


_CODE_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)


def _extract_code(llm_output: str) -> str:
    """Pull the Python code out of a fenced code block. Falls back to
    treating the whole output as code if the model forgot the fences."""
    match = _CODE_BLOCK_RE.search(llm_output)
    if match:
        return match.group(1).strip()
    return llm_output.strip()


def _summarize_result(value: Any, stdout: str) -> str:
    """Compact repr of a result for the explanation LLM.
    We truncate large frames so the prompt stays small."""
    if value is None and stdout:
        return f"stdout: {stdout.strip()[:500]}"
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "Empty DataFrame (0 rows matched)."
        return (
            f"DataFrame with {len(value)} rows and {len(value.columns)} cols:\n"
            f"{value.head(10).to_string(max_cols=8, max_colwidth=30)}"
        )
    if isinstance(value, pd.Series):
        if value.empty:
            return "Empty Series (0 values)."
        return f"Series ({len(value)} entries):\n{value.head(10).to_string()}"
    return f"{type(value).__name__}: {repr(value)[:500]}"


def _generate_explanation(question: str, code: str, value: Any, stdout: str) -> str:
    """Ask the LLM to explain the result in plain English."""
    try:
        result_repr = _summarize_result(value, stdout)
        messages = build_explain_messages(question, code, result_repr)
        llm = _get_llm()
        lc_messages = [
            SystemMessage(content=messages[0]["content"]),
            HumanMessage(content=messages[1]["content"]),
        ]
        response = llm.invoke(lc_messages)
        return response.content.strip()
    except Exception:
        # Explanation is nice-to-have — never let it break the whole flow
        return ""


def ask(
    question: str,
    datasets: dict[str, Dataset] | Dataset,
    history: list[QATurn] | None = None,
    primary_name: str | None = None,
) -> QATurn:
    """Ask Vera a question.

    Accepts either a single Dataset (backward-compat) OR a dict of
    {variable_name: Dataset} for multi-file Q&A. In multi-file mode,
    every dataframe is exposed in the sandbox by its variable name,
    so the LLM can `.merge()` across them.

    Args:
        question: user's natural-language question
        datasets: one Dataset, or {var_name: Dataset} mapping
        history: prior QATurns for context
        primary_name: which df's head to show in the prompt (defaults
                      to the first key in `datasets`)
    """
    history = history or []

    # Backward-compat: single Dataset becomes a one-entry dict
    if isinstance(datasets, Dataset):
        datasets = {"df": datasets}

    if primary_name is None or primary_name not in datasets:
        primary_name = next(iter(datasets))

    primary_ds = datasets[primary_name]
    dataframes = {name: ds.df for name, ds in datasets.items()}

    schema = _format_schema(datasets)
    head = primary_ds.df.head(3).to_string(max_cols=10, max_colwidth=30)
    history_str = _format_history(history)

    messages = build_qa_messages(
        question=question,
        schema=schema,
        head=head,
        n_dataframes=len(datasets),
        primary_name=primary_name,
        history=history_str,
    )

    llm = _get_llm()
    lc_messages = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ]

    response = llm.invoke(lc_messages)
    raw = response.content
    code = _extract_code(raw)

    # Handle the LLM's escape hatch
    if "CANNOT_ANSWER" in code:
        return QATurn(
            question=question,
            code=code,
            success=False,
            error="Vera couldn't answer this from the available data.",
        )

    # Execute with all dataframes in scope
    result = safe_execute(code, dataframes=dataframes)

    explanation = ""
    if result.success:
        explanation = _generate_explanation(question, code, result.value, result.stdout)

    return QATurn(
        question=question,
        code=code,
        success=result.success,
        value=result.value,
        figure=result.figure,
        error=result.error,
        stdout=result.stdout,
        explanation=explanation,
    )