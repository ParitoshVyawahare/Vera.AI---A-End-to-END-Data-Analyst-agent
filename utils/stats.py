"""Statistical summaries, correlations, outliers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class OutlierReport:
    column: str
    method: str
    n_outliers: int
    pct: float
    bounds: tuple[float, float] | None = None


def summary_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc["skew"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    desc["missing"] = df[numeric_cols].isna().sum()
    desc["missing_pct"] = (100 * desc["missing"] / len(df)).round(2)
    return desc.round(3)


def categorical_summary(df: pd.DataFrame, cat_cols: list[str], top_n: int = 5) -> dict[str, pd.DataFrame]:
    out = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(top_n).reset_index()
        vc.columns = [c, "count"]
        vc["pct"] = (100 * vc["count"] / len(df)).round(2)
        out[c] = vc
    return out


def top_correlations(df: pd.DataFrame, numeric_cols: list[str], top_k: int = 10, min_abs: float = 0.3) -> pd.DataFrame:
    if len(numeric_cols) < 2:
        return pd.DataFrame(columns=["col_a", "col_b", "correlation"])
    corr = df[numeric_cols].corr(numeric_only=True).abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    pairs.columns = ["col_a", "col_b", "correlation"]
    signed = df[numeric_cols].corr(numeric_only=True)
    pairs["correlation"] = pairs.apply(lambda r: signed.loc[r.col_a, r.col_b], axis=1).round(3)
    return (
        pairs[pairs["correlation"].abs() >= min_abs]
        .reindex(pairs["correlation"].abs().sort_values(ascending=False).index)
        .head(top_k)
        .reset_index(drop=True)
    )


def outliers_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> list[OutlierReport]:
    reports = []
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        n = int(mask.sum())
        if n == 0:
            continue
        reports.append(OutlierReport(
            column=c, method="IQR", n_outliers=n,
            pct=round(100 * n / len(df), 2),
            bounds=(round(float(low), 3), round(float(high), 3)),
        ))
    return reports


def outliers_isolation_forest(df: pd.DataFrame, numeric_cols: list[str], contamination: float = 0.05) -> OutlierReport | None:
    if len(numeric_cols) < 2:
        return None
    X = df[numeric_cols].dropna()
    if len(X) < 20:
        return None
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    n = int((preds == -1).sum())
    return OutlierReport(
        column=f"[multivariate: {', '.join(numeric_cols)}]",
        method="IsolationForest",
        n_outliers=n,
        pct=round(100 * n / len(X), 2),
    )


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing": int(df.isna().sum().sum()),
        "cols_with_missing": int((df.isna().sum() > 0).sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }