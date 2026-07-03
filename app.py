"""Vera.AI — Streamlit entry point."""

import streamlit as st
import pandas as pd
from pathlib import Path

from utils.data_loader import load_file, sanitize_name
from agents.analyzer import analyze
from agents.report_agent import stream_report
from agents.chart_agent import generate_charts
from agents.qa_agent import ask, QATurn
from utils.exporters import build_markdown, build_pdf, build_bundle_zip, timestamp

st.set_page_config(
    page_title="Vera.AI — analysis you can cite",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  html, body, [class*="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #FBFAF7;
    color: #1A1A1A;
  }
  .main .block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
  }
  .vera-hero {
    text-align: center;
    padding: 2rem 0 1.5rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid #E8E4D9;
  }
  .vera-wordmark {
    font-family: 'Instrument Serif', Georgia, serif;
    font-size: 4rem;
    font-weight: 400;
    line-height: 1;
    letter-spacing: -0.02em;
    color: #1A1A1A;
    margin: 0;
  }
  .vera-wordmark .accent {
    color: #D97757;
    font-style: italic;
  }
  .vera-tagline {
    font-family: 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-size: 1.2rem;
    color: #6B6B6B;
    margin-top: 0.6rem;
    letter-spacing: 0.01em;
  }
  #MainMenu, footer, header {visibility: hidden;}
  .stDeployButton {display: none;}
  .vera-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6B6B6B;
    margin: 2rem 0 0.6rem 0;
  }
  [data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #D9D4C4;
    border-radius: 10px;
    padding: 1rem;
  }
  [data-testid="stFileUploader"]:hover { border-color: #0E5C4A; }
  [data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
  }
  [data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E4D9;
    border-radius: 10px;
    padding: 1rem 1.25rem;
  }
  [data-testid="stMetric"]:hover { border-color: #0E5C4A; }
  [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #6B6B6B !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 1.75rem !important;
    font-weight: 500 !important;
    color: #0E5C4A !important;
    letter-spacing: -0.02em !important;
  }
  .stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #E8E4D9;
  }
  .stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 1.1rem;
    background: transparent;
    color: #6B6B6B;
    font-weight: 500;
    font-size: 0.92rem;
    border-radius: 0;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
  }
  .stTabs [aria-selected="true"] {
    color: #0E5C4A !important;
    border-bottom: 2px solid #D97757 !important;
    background: transparent !important;
  }
  .stButton > button {
    background: #FFFFFF;
    color: #1A1A1A;
    border: 1px solid #E8E4D9;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-weight: 500;
    font-size: 0.9rem;
    transition: all 0.15s ease;
    text-align: left;
    line-height: 1.4;
  }
  .stButton > button:hover {
    border-color: #0E5C4A;
    background: #F3F1EA;
    color: #0E5C4A;
    transform: translateY(-1px);
  }
  .stButton > button[kind="primary"] {
    background: #0E5C4A;
    color: white;
    border: 1px solid #0E5C4A;
  }
  .stButton > button[kind="primary"]:hover {
    background: #0A4A3B;
    border-color: #0A4A3B;
    color: white;
  }
  [data-testid="stDownloadButton"] > button {
    background: #FFFFFF;
    color: #0E5C4A;
    border: 1px solid #0E5C4A;
    text-align: center;
    justify-content: center;
  }
  [data-testid="stDownloadButton"] > button:hover {
    background: #0E5C4A;
    color: white;
  }
  [data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E8E4D9;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  [data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #E8E4D9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 1rem !important;
  }
  [data-testid="stChatInput"] textarea:focus {
    border-color: #0E5C4A !important;
    box-shadow: 0 0 0 3px rgba(14, 92, 74, 0.1) !important;
  }
  code, pre, [data-testid="stCodeBlock"] {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stDataFrame"] {
    border: 1px solid #E8E4D9;
    border-radius: 8px;
    overflow: hidden;
  }
  .streamlit-expanderHeader {
    background: #F3F1EA;
    border-radius: 8px;
    font-weight: 500;
  }
  [data-testid="stAlert"] {
    border-radius: 10px;
    border-left: 3px solid;
  }
  hr { border-color: #E8E4D9 !important; }
  .vera-footer {
    text-align: center;
    color: #6B6B6B;
    font-family: 'Instrument Serif', Georgia, serif;
    font-style: italic;
    font-size: 0.95rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #E8E4D9;
  }
</style>
""", unsafe_allow_html=True)


# ─── Hero header ────────────────────────────────────────────────
st.markdown("""
<div class="vera-hero">
  <div class="vera-wordmark">Vera<span class="accent">.AI</span></div>
  <div class="vera-tagline">the analyst that only speaks in numbers</div>
</div>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────
if "datasets" not in st.session_state:
    st.session_state.datasets = {}          # dict[var_name -> Dataset]
if "primary_name" not in st.session_state:
    st.session_state.primary_name = None    # which df is primary
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None
if "charts" not in st.session_state:
    st.session_state.charts = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ─── Upload ───────────────────────────────────────────────────────
st.markdown('<div class="vera-label">Upload your dataset(s)</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    label="Upload your dataset",
    type=["csv", "tsv", "xlsx", "xls", "parquet", "json"],
    label_visibility="collapsed",
    accept_multiple_files=True,
    help="Upload one file, or multiple to enable cross-file Q&A (merge/join).",
)

# Detect if the set of uploaded files has changed
new_names = tuple(sorted(f.name for f in uploaded_files)) if uploaded_files else ()
existing_names = tuple(sorted(ds.source_name for ds in st.session_state.datasets.values()))

if uploaded_files and new_names != existing_names:
    with st.spinner(f"Reading and profiling {len(uploaded_files)} file(s)..."):
        new_datasets = {}
        for f in uploaded_files:
            var_name = sanitize_name(f.name)
            # Handle collisions (two files sanitize to the same key)
            suffix = 2
            base = var_name
            while var_name in new_datasets:
                var_name = f"{base}_{suffix}"
                suffix += 1
            new_datasets[var_name] = load_file(f)

        st.session_state.datasets = new_datasets
        st.session_state.primary_name = next(iter(new_datasets))
        primary_ds = new_datasets[st.session_state.primary_name]
        st.session_state.analysis = analyze(primary_ds)
        st.session_state.report_md = None
        st.session_state.charts = generate_charts(primary_ds, st.session_state.analysis)
        st.session_state.qa_history = []

# Nothing uploaded — show prompt and stop
if not st.session_state.datasets:
    st.markdown("""
    <div style="text-align:center; color:#6B6B6B; font-family:'Instrument Serif', Georgia, serif;
                font-style:italic; font-size:1.05rem; margin-top:2rem;">
      Drop a file above to begin.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Primary-file picker (multi-file mode only) ──────────────────
if len(st.session_state.datasets) > 1:
    st.markdown('<div class="vera-label">Primary dataset</div>', unsafe_allow_html=True)
    col_pick, col_info = st.columns([2, 3])
    with col_pick:
        keys = list(st.session_state.datasets.keys())
        selected = st.selectbox(
            "Primary dataset",
            options=keys,
            index=keys.index(st.session_state.primary_name),
            format_func=lambda k: f"{k}  ({st.session_state.datasets[k].source_name})",
            label_visibility="collapsed",
        )
    with col_info:
        st.caption(
            f"💬 Ask Vera can query all {len(st.session_state.datasets)} files. "
            f"Reports, charts, and overview use the primary."
        )

    if selected != st.session_state.primary_name:
        with st.spinner("Re-analyzing with new primary dataset..."):
            st.session_state.primary_name = selected
            primary_ds = st.session_state.datasets[selected]
            st.session_state.analysis = analyze(primary_ds)
            st.session_state.report_md = None
            st.session_state.charts = generate_charts(primary_ds, st.session_state.analysis)
        st.rerun()

# Primary dataset for report/charts/overview
ds = st.session_state.datasets[st.session_state.primary_name]
res = st.session_state.analysis


# ─── Quality warnings (across all uploaded files) ────────────────
for var_name, dataset in st.session_state.datasets.items():
    if dataset.renamed_columns:
        renames = ", ".join(f"`{orig}` → `{new}`" for new, orig in dataset.renamed_columns.items())
        st.warning(f"⚠️ `{dataset.source_name}`: duplicate column names renamed — {renames}")


# ─── Overview metrics ────────────────────────────────────────────
st.markdown('<div class="vera-label">Overview</div>', unsafe_allow_html=True)
q = res.quality
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{q['n_rows']:,}")
c2.metric("Columns", q["n_cols"])
c3.metric("Missing cells", f"{q['total_missing']:,}")
c4.metric("Duplicate rows", q["duplicate_rows"])
c5.metric("Memory (MB)", q["memory_mb"])


# ─── Answer rendering helper ─────────────────────────────────────
def _render_answer(turn: QATurn):
    """Render a QATurn's answer inline."""
    if not turn.success:
        st.error(f"⚠️ {turn.error}")
        with st.expander("View generated code"):
            st.code(turn.code, language="python")
        return

    if turn.explanation:
        st.markdown(
            f"<div style='font-size:1.02rem; color:#1A1A1A; margin-bottom:0.9rem; "
            f"line-height:1.55;'>{turn.explanation}</div>",
            unsafe_allow_html=True,
        )

    if turn.figure is not None:
        st.pyplot(turn.figure)
    elif isinstance(turn.value, (pd.DataFrame, pd.Series)):
        st.dataframe(turn.value, use_container_width=True)
    elif turn.value is not None:
        if not turn.explanation:
            st.write(turn.value)
    elif turn.stdout:
        if not turn.explanation:
            st.write(turn.stdout.strip())
    elif not turn.explanation:
        st.info("Vera returned no output.")

    with st.expander("View generated code"):
        st.code(turn.code, language="python")


# ─── Report download buttons helper ──────────────────────────────
def _render_download_buttons():
    """Show three download buttons: Markdown, PDF, Bundle."""
    st.markdown('<div class="vera-label" style="margin-top:2rem;">Export</div>', unsafe_allow_html=True)
    base_name = Path(ds.source_name).stem
    ts = timestamp()

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            label="📥 Download Markdown",
            data=build_markdown(st.session_state.report_md, ds.source_name),
            file_name=f"vera_report_{base_name}_{ts}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with dl2:
        try:
            pdf_bytes = build_pdf(st.session_state.report_md, ds.source_name)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name=f"vera_report_{base_name}_{ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception:
            st.button("📄 PDF (unavailable)", disabled=True, use_container_width=True)

    with dl3:
        try:
            bundle_bytes = build_bundle_zip(
                st.session_state.report_md,
                ds.source_name,
                res.to_context_dict(),
                st.session_state.charts,
            )
            st.download_button(
                label="📦 Download Bundle (.zip)",
                data=bundle_bytes,
                file_name=f"vera_bundle_{base_name}_{ts}.zip",
                mime="application/zip",
                use_container_width=True,
                help="Report (MD + PDF) + all charts as PNG + raw analysis as JSON",
            )
        except Exception:
            st.button("📦 Bundle (unavailable)", disabled=True, use_container_width=True)


# ─── Tabs ────────────────────────────────────────────────────────
st.markdown('<div class="vera-label" style="margin-top:2.5rem;">Analysis</div>', unsafe_allow_html=True)

tab_qa, tab_report, tab_charts, tab_preview, tab_schema, tab_stats, tab_corr, tab_outliers = st.tabs(
    ["Ask Vera", "AI Report", "Charts", "Preview", "Schema", "Summary", "Correlations", "Outliers"]
)

with tab_qa:
    n_files = len(st.session_state.datasets)
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
      <div style="font-family:'Instrument Serif',Georgia,serif; font-size:1.6rem; margin-bottom:0.4rem;">
        Ask Vera about your data
      </div>
      <div style="color:#6B6B6B; font-size:0.95rem;">
        Vera writes pandas code to answer your questions, runs it safely, and returns the result.
        {"She can query across all " + str(n_files) + " uploaded files." if n_files > 1 else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.qa_history:
        st.markdown('<div class="vera-label">Try one of these</div>', unsafe_allow_html=True)

        # Suggestions adapt to single vs multi-file mode
        if n_files > 1:
            df_names = list(st.session_state.datasets.keys())
            suggestions = [
                f"How many rows are in each dataframe?",
                f"What columns do {df_names[0]} and {df_names[1]} have in common?",
                f"What are the column types across all dataframes?",
                f"Show the summary statistics for {df_names[0]}",
            ]
        else:
            suggestions = [
                "How many rows and columns are there?",
                "What are the column names and their types?",
                "Show the top 5 rows sorted by the first numeric column",
                "What are the summary statistics of the numeric columns?",
            ]

        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"suggest_{i}", use_container_width=True):
                st.session_state._pending_question = s
                st.rerun()

    for turn in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant", avatar="📊"):
            _render_answer(turn)

    pending = st.session_state.pop("_pending_question", None)
    user_q = st.chat_input("Ask about your data...")
    question = user_q or pending

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant", avatar="📊"):
            with st.spinner("Vera is thinking..."):
                try:
                    turn = ask(
                        question,
                        st.session_state.datasets,
                        history=st.session_state.qa_history,
                        primary_name=st.session_state.primary_name,
                    )
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    turn = None

            if turn:
                _render_answer(turn)
                st.session_state.qa_history.append(turn)


with tab_report:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-family:'Instrument Serif',Georgia,serif; font-size:1.6rem; margin-bottom:0.4rem;">
        AI-Generated Analysis Report
      </div>
      <div style="color:#6B6B6B; font-size:0.95rem;">
        A structured briefing written by Vera — grounded strictly in your data.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 5])
    with col_a:
        generate_clicked = st.button("Generate report", type="primary")
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
            _render_download_buttons()
        except Exception as e:
            st.error(f"Report generation failed: {e}")
    elif st.session_state.report_md:
        st.markdown(st.session_state.report_md)
        _render_download_buttons()
    else:
        st.info("Click **Generate report** to have Vera analyze this dataset.")


with tab_charts:
    st.markdown("""
    <div style="margin-bottom:1.5rem;">
      <div style="font-family:'Instrument Serif',Georgia,serif; font-size:1.6rem; margin-bottom:0.4rem;">
        Auto-generated visualizations
      </div>
    </div>
    """, unsafe_allow_html=True)
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
        st.markdown('<div class="vera-label">Numeric columns</div>', unsafe_allow_html=True)
        st.dataframe(res.summary_numeric, use_container_width=True)
    if res.summary_categorical:
        st.markdown('<div class="vera-label">Categorical columns</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="vera-label">Per-column (IQR method)</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([o.__dict__ for o in res.outliers_univariate]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.write("No univariate outliers found.")
    if res.outliers_multivariate:
        st.markdown('<div class="vera-label">Multivariate (Isolation Forest)</div>', unsafe_allow_html=True)
        st.json(res.outliers_multivariate.__dict__)


# ─── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="vera-footer">
  Vera.AI — analysis you can cite.
</div>
""", unsafe_allow_html=True)