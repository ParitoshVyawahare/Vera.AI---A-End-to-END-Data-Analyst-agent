"""Centralized prompt templates. All prompt engineering lives here."""

from __future__ import annotations

# ─── Report agent prompts ────────────────────────────────────────────
REPORT_SYSTEM = """You are a senior data analyst writing a briefing for a \
business stakeholder. You are precise, quantitative, and never invent facts. \
You only reference numbers that appear in the provided context. If something \
is not in the context, you do not mention it. You write in clear prose, not \
bullet-heavy filler. Every claim must be traceable to the data."""


REPORT_USER = """Below is an automated analysis of a dataset called \
`{source_name}`. Write a structured report in Markdown with these sections:

## Executive Summary
Two or three sentences: what this dataset is, its size, and the single most \
important thing a decision-maker should know.

## Data Quality
Comment on missing values, duplicates, and any columns that look problematic. \
Be specific — name the columns and cite the percentages.

## Key Findings
Three to five findings. Each finding is one short paragraph. Prioritize \
findings involving correlations, notable distributions, or outliers. \
Reference exact column names in backticks and cite numbers.

## Anomalies & Outliers
Interpret the outlier detection results. Which columns have the most outliers? \
Does the multivariate detector (Isolation Forest) agree with the univariate \
one (IQR)? What might explain them — without speculating beyond the data?

## Recommended Next Questions
List 5 specific analytical questions a stakeholder should investigate next, \
based on what you actually see in this data. Phrase them so they could be \
answered by querying the dataframe. Number them 1-5.

---

DATASET CONTEXT:
{context}
"""


def build_report_messages(source_name: str, context: str) -> list[dict]:
    """Build the message list for the report LLM call."""
    return [
        {"role": "system", "content": REPORT_SYSTEM},
        {"role": "user", "content": REPORT_USER.format(
            source_name=source_name,
            context=context,
        )},
    ]