"""Orchestrates the auto-analysis pass over a Dataset."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

from utils.data_loader import Dataset
from utils.stats import (
    summary_stats,
    categorical_summary,
    top_correlations,
    outliers_iqr,
    outliers_isolation_forest,
    quality_report,
)


@dataclass
class AnalysisResult:
    quality: dict
    summary_numeric: pd.DataFrame
    summary_categorical: dict
    correlations: pd.DataFrame
    outliers_univariate: list
    outliers_multivariate: object | None
    profiles: list

    def to_context_dict(self) -> dict:
        return {
            "quality": self.quality,
            "profiles": [asdict(p) for p in self.profiles],
            "summary_numeric": self.summary_numeric.to_dict() if not self.summary_numeric.empty else {},
            "correlations": self.correlations.to_dict(orient="records"),
            "outliers_univariate": [asdict(o) for o in self.outliers_univariate],
            "outliers_multivariate": asdict(self.outliers_multivariate) if self.outliers_multivariate else None,
        }


def analyze(dataset: Dataset) -> AnalysisResult:
    df = dataset.df
    numeric_cols = dataset.columns_by_kind("numeric")
    cat_cols = dataset.columns_by_kind("categorical")

    return AnalysisResult(
        quality=quality_report(df),
        summary_numeric=summary_stats(df, numeric_cols),
        summary_categorical=categorical_summary(df, cat_cols),
        correlations=top_correlations(df, numeric_cols),
        outliers_univariate=outliers_iqr(df, numeric_cols),
        outliers_multivariate=outliers_isolation_forest(df, numeric_cols),
        profiles=dataset.profiles,
    )