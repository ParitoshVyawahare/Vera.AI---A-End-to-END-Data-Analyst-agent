"""AI Data Analyst — Streamlit entry point."""

import streamlit as st
import pandas as pd

from utils.data_loader import load_file
from agents.analyzer import analyze
from agents.report_agent import stream_report
from agents.chart_agent import generate_charts

st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="📊")
st.title("📊 AI Data Analyst")
st.caption("Upload a dataset. Get automatic analysis, charts, and a written report. Then ask questions.")

# ─── Session state ────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None
if "charts" not in st.session_state:
    st.session_state.charts = None

# ─── Upload ───────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop a CSV, Excel, Parquet, or JSON file",
    type=["csv", "tsv", "xlsx", "xls", "parquet", "json"],
)

if uploaded is not None and (
    st.session_state.dataset is None
    or st.session_state.dataset.source_name != uploaded.name
):
    with st.spinner("Reading and profiling your data..."):
        st.session_state.dataset = load_file(uploaded)
        st.session_state.analysis = analyze(st.session_state.dataset)
        st.session_state.report_md = None
        st.session_state.charts = generate_charts(
            st.session_state.dataset, st.session_state.analysis
        )

# ─── Render ───────────────────────────────────────────────────────
ds = st.session_state.dataset
res = st.session_state.analysis

if ds is None:
    st.info("👆 Upload a file to begin.")
    st.stop()

# Data quality heads-up
if ds.renamed_columns:
    renames = ", ".join(f"`{orig}` → `{new}`" for new, orig in ds.renamed_columns.items())
    st.warning(f"⚠️ Duplicate column names detected and renamed: {renames}")
# Overview strip
q = res.quality
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{q['n_rows']:,}")
c2.metric("Columns", q["n_cols"])
c3.metric("Missing cells", f"{q['total_missing']:,}")
c4.metric("Duplicate rows", q["duplicate_rows"])
c5.metric("Memory (MB)", q["memory_mb"])

tab_report, tab_charts, tab_preview, tab_schema, tab_stats, tab_corr, tab_outliers = st.tabs(
    ["🤖 AI Report", "📈 Charts", "Preview", "Schema", "Summary stats", "Correlations", "Outliers"]
)

with tab_report:
    st.subheader("AI-Generated Analysis Report")

    col_a, col_b = st.columns([1, 5])
    with col_a:
        generate_clicked = st.button("✨ Generate report", type="primary")
    with col_b:
        if st.session_state.report_md:
            st.caption("Report ready — click again to regenerate.")

    if generate_clicked:
        try:
            report_placeholder = st.empty()
            streamed = report_placeholder.write_stream(
                stream_report(res, ds.source_name)
            )
            st.session_state.report_md = streamed
        except Exception as e:
            st.error(f"Report generation failed: {e}")
    elif st.session_state.report_md:
        st.markdown(st.session_state.report_md)
    else:
        st.info("Click **Generate report** to have the AI analyze this dataset.")

with tab_charts:
    st.subheader("Auto-generated visualizations")
    if not st.session_state.charts:
        st.info("No charts to display for this dataset.")
    else:
        for i in range(0, len(st.session_state.charts), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(st.session_state.charts):
                    chart = st.session_state.charts[i + j]
                    with col:
                        st.plotly_chart(chart.figure, use_container_width=True)
                        if chart.caption:
                            st.caption(chart.caption)

with tab_preview:
    st.dataframe(ds.df.head(50), use_container_width=True)

with tab_schema:
    schema_df = pd.DataFrame([{
        "column": p.name,
        "inferred_kind": p.kind,
        "dtype": p.dtype,
        "n_unique": p.n_unique,
        "missing_%": p.missing_pct,
        "sample": ", ".join(str(v) for v in p.sample_values),
    } for p in res.profiles])
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

with tab_stats:
    if not res.summary_numeric.empty:
        st.subheader("Numeric columns")
        st.dataframe(res.summary_numeric, use_container_width=True)
    if res.summary_categorical:
        st.subheader("Categorical columns (top values)")
        for col, vc in res.summary_categorical.items():
            with st.expander(col):
                st.dataframe(vc, use_container_width=True, hide_index=True)

with tab_corr:
    if res.correlations.empty:
        st.write("No strong correlations detected (|r| ≥ 0.3).")
    else:
        st.dataframe(res.correlations, use_container_width=True, hide_index=True)

with tab_outliers:
    if res.outliers_univariate:
        st.subheader("Per-column (IQR method)")
        st.dataframe(
            pd.DataFrame([o.__dict__ for o in res.outliers_univariate]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No univariate outliers found.")
    if res.outliers_multivariate:
        st.subheader("Multivariate (Isolation Forest)")
        st.json(res.outliers_multivariate.__dict__)

st.divider()
st.caption("Next up: Q&A chat with sandboxed code execution.")