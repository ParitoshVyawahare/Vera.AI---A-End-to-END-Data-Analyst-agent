"""Auto-generates Plotly charts based on the Dataset's column profile."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import Dataset
from agents.analyzer import AnalysisResult

ChartKind = Literal["histogram", "bar", "heatmap", "timeseries", "scatter"]


@dataclass
class Chart:
    title: str
    kind: ChartKind
    figure: go.Figure
    caption: str = ""


_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
    height=380,
    title_font_size=15,
)


def _apply_style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _histogram(series: pd.Series) -> go.Figure:
    fig = px.histogram(series.dropna(), nbins=30, opacity=0.85)
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    fig.update_layout(showlegend=False, xaxis_title=series.name, yaxis_title="count")
    return fig


def _bar_top_values(series: pd.Series, top_n: int = 10) -> go.Figure:
    vc = series.value_counts(dropna=False).head(top_n).sort_values()
    fig = px.bar(
        x=vc.values, y=vc.index.astype(str),
        orientation="h", labels={"x": "count", "y": series.name},
    )
    fig.update_layout(showlegend=False)
    return fig


def _corr_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
    corr = df[numeric_cols].corr(numeric_only=True).round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    )
    fig.update_layout(coloraxis_colorbar_title="r")
    return fig


def _timeseries(df: pd.DataFrame, date_col: str, value_col: str) -> go.Figure:
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .sort_values(date_col)
        .groupby(pd.Grouper(key=date_col, freq="D"))[value_col]
        .mean()
        .reset_index()
    )
    fig = px.line(ts, x=date_col, y=value_col)
    fig.update_traces(line_width=2)
    return fig


def _scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    sub = df[[x, y]].copy()
    sub = _ensure_unique_columns(sub)
    fig = px.scatter(
        sub, x=x, y=y, opacity=0.6, trendline="ols",
        trendline_color_override="crimson",
    )
    return fig


def _pick_top_numerics(df: pd.DataFrame, numeric_cols: list[str], k: int = 6) -> list[str]:
    numeric_cols = list(dict.fromkeys(numeric_cols))
    if len(numeric_cols) <= k:
        return numeric_cols
    scores = {}
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) == 0 or s.mean() == 0:
            scores[c] = 0
            continue
        scores[c] = abs(s.std() / s.mean())
    return sorted(scores, key=scores.get, reverse=True)[:k]


def _pick_top_categoricals(df: pd.DataFrame, cat_cols: list[str], k: int = 6) -> list[str]:
    cat_cols = list(dict.fromkeys(cat_cols))
    scored = []
    for c in cat_cols:
        n_unique = df[c].nunique(dropna=True)
        if 2 <= n_unique <= 30:
            scored.append((c, n_unique))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:k]]


def generate_charts(dataset: Dataset, analysis: AnalysisResult) -> list[Chart]:
    df = _ensure_unique_columns(dataset.df)
    numeric_cols = [c for c in dataset.columns_by_kind("numeric") if c in df.columns]
    cat_cols = [c for c in dataset.columns_by_kind("categorical") if c in df.columns]
    dt_cols = [c for c in dataset.columns_by_kind("datetime") if c in df.columns]

    charts: list[Chart] = []

    if len(numeric_cols) >= 2:
        fig = _apply_style(_corr_heatmap(df, numeric_cols), "Correlation heatmap")
        charts.append(Chart(
            title="Correlation heatmap",
            kind="heatmap",
            figure=fig,
            caption=f"Pairwise Pearson correlations across {len(numeric_cols)} numeric columns.",
        ))

    if dt_cols and numeric_cols:
        date_col = dt_cols[0]
        for value_col in _pick_top_numerics(df, numeric_cols, k=3):
            fig = _apply_style(
                _timeseries(df, date_col, value_col),
                f"{value_col} over time",
            )
            charts.append(Chart(
                title=f"{value_col} over time",
                kind="timeseries",
                figure=fig,
                caption=f"Daily mean of `{value_col}` by `{date_col}`.",
            ))

    if not analysis.correlations.empty:
        strong = analysis.correlations[
            analysis.correlations["correlation"].abs() >= 0.7
        ].head(3)
        for _, row in strong.iterrows():
            col_a, col_b = row["col_a"], row["col_b"]
            if col_a == col_b or col_a not in df.columns or col_b not in df.columns:
                continue
            fig = _apply_style(
                _scatter(df, col_a, col_b),
                f"{col_a} vs {col_b}  (r = {row['correlation']:.2f})",
            )
            charts.append(Chart(
                title=f"{col_a} vs {col_b}",
                kind="scatter",
                figure=fig,
                caption=f"Pearson r = {row['correlation']:.2f}. OLS trend line overlaid.",
            ))

    for col in _pick_top_numerics(df, numeric_cols, k=6):
        fig = _apply_style(_histogram(df[col]), f"Distribution of {col}")
        charts.append(Chart(
            title=f"Distribution of {col}",
            kind="histogram",
            figure=fig,
            caption=f"Skew = {df[col].skew():.2f}, Kurtosis = {df[col].kurtosis():.2f}",
        ))

    for col in _pick_top_categoricals(df, cat_cols, k=6):
        fig = _apply_style(_bar_top_values(df[col]), f"Top values in {col}")
        charts.append(Chart(
            title=f"Top values in {col}",
            kind="bar",
            figure=fig,
            caption=f"Top 10 most frequent values ({df[col].nunique()} unique total).",
        ))

    return charts
