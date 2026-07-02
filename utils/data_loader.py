"""Load tabular files and infer semantic column types."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import re
import warnings
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
    renamed_columns: dict[str, str] = field(default_factory=dict)

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

    df, renamed = _deduplicate_columns(df)
    df = _fix_fake_datetimes(df)
    df = _try_parse_datetimes(df)
    profiles = [_profile_column(df[c]) for c in df.columns]
    return Dataset(df=df, profiles=profiles, source_name=name, renamed_columns=renamed)


def _deduplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename duplicate columns by appending _2, _3, etc.
    Real-world CSVs (Excel exports, bad joins) often have duplicate headers.
    Returns the new dataframe and a map of {new_name: original_name}."""
    cols = pd.Series(df.columns)
    duplicates = cols[cols.duplicated()].unique()
    renamed: dict[str, str] = {}
    for dup in duplicates:
        idx = cols[cols == dup].index.tolist()
        # Leave the first occurrence, rename the rest: bpm, bpm_2, bpm_3, ...
        for i, pos in enumerate(idx[1:], start=2):
            new_name = f"{dup}_{i}"
            cols.iloc[pos] = new_name
            renamed[new_name] = dup
    df.columns = cols.tolist()
    return df, renamed


def _fix_fake_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Undo pandas' auto-parsing of pure-time columns (HH:MM:SS with no date).
    When a CSV has values like '08:57:13', pandas prepends TODAY's date to
    make them full datetimes — producing datetime64 columns that are actually
    time-of-day, not real timestamps. We detect these (all values share the
    same date) and convert them back to strings so downstream agents treat
    them as text rather than misleading datetimes."""
    for col in df.select_dtypes(include="datetime64").columns:
        s = df[col].dropna()
        if s.empty:
            continue
        # If every value falls on the same calendar day, this is a fake datetime
        if s.dt.normalize().nunique() == 1:
            df[col] = df[col].dt.strftime("%H:%M:%S")
    return df


def _try_parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to parse object columns that look like dates.
    Only object-dtype columns are considered, and we require the values to
    contain typical date-like characters (- / or a 4-digit year). This
    avoids misclassifying columns like '14.95' or '08:57:13' as datetimes."""
    date_pattern = re.compile(r"[-/]|\b\d{4}\b")

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(20)
        if sample.empty:
            continue

        # Require most sampled values to look date-ish
        looks_dateish = sample.apply(lambda x: bool(date_pattern.search(x))).mean()
        if looks_dateish < 0.8:
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