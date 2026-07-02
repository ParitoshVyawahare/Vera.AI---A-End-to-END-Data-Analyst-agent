"""Load tabular files and infer semantic column types."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np

ColumnKind = Literal["numeric", "categorical", "datetime", "boolean", "text", "id"]


@dataclass
class ColumnProfile:
    name: str
    kind: ColumnKind
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    sample_values: list = field(default_factory=list)


@dataclass
class Dataset:
    df: pd.DataFrame
    profiles: list[ColumnProfile]
    source_name: str

    def columns_by_kind(self, kind: ColumnKind) -> list[str]:
        return [p.name for p in self.profiles if p.kind == kind]


def load_file(uploaded_file, filename: str | None = None) -> Dataset:
    """Load a Streamlit UploadedFile (or path) into a Dataset."""
    name = filename or getattr(uploaded_file, "name", "data")
    ext = Path(name).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(uploaded_file)
    elif ext == ".tsv":
        df = pd.read_csv(uploaded_file, sep="\t")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded_file)
    elif ext == ".parquet":
        df = pd.read_parquet(uploaded_file)
    elif ext == ".json":
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df = _try_parse_datetimes(df)
    profiles = [_profile_column(df[c]) for c in df.columns]
    return Dataset(df=df, profiles=profiles, source_name=name)


def _try_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to parse object columns that look like dates.
    Suppresses the noisy 'could not infer format' warning — falling back
    to dateutil is expected behavior for our detection heuristic."""
    import warnings
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(20)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                pd.to_datetime(sample, errors="raise")
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except (ValueError, TypeError):
                continue
    return df


def _profile_column(s: pd.Series) -> ColumnProfile:
    n = len(s)
    n_missing = int(s.isna().sum())
    n_unique = int(s.nunique(dropna=True))
    dtype = str(s.dtype)
    kind = _infer_kind(s, n_unique, n)
    sample = s.dropna().unique()[:5].tolist()
    return ColumnProfile(
        name=str(s.name),
        kind=kind,
        dtype=dtype,
        n_unique=n_unique,
        n_missing=n_missing,
        missing_pct=round(100 * n_missing / max(n, 1), 2),
        sample_values=[_json_safe(v) for v in sample],
    )


def _infer_kind(s: pd.Series, n_unique: int, n: int) -> ColumnKind:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        if n_unique <= 10 and n_unique / max(n, 1) < 0.05:
            return "categorical"
        return "numeric"
    if n_unique == n and n > 20:
        return "id"
    if n_unique / max(n, 1) < 0.5 and n_unique <= 50:
        return "categorical"
    return "text"


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    return v