<div align="center">

# 📊 Vera.AI

### *the analyst that only speaks in numbers*

**An AI data analyst that reads your CSV, writes a full report, generates charts, and answers your questions in plain English — grounded strictly in your data, never hallucinated.**

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-D97757?style=for-the-badge)](https://huggingface.co/spaces/Paritosh271201/vera-ai)
[![Python](https://img.shields.io/badge/Python-3.11-0E5C4A?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-D97757?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Llama_3.3_70B-0E5C4A?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-0E5C4A?style=for-the-badge)](LICENSE)

[**🚀 Try Vera Live**](https://huggingface.co/spaces/Paritosh271201/vera-ai) · [**📖 Architecture**](#-architecture) · [**⚡ Features**](#-features) · [**🛠️ Tech Stack**](#-tech-stack)

</div>

---

## 🎯 What is Vera?

Most AI data tools **hallucinate** — inventing trends, misquoting numbers, guessing at columns that don't exist. Vera is the opposite: her system prompt forbids any claim not traceable to your dataset, and her Q&A agent executes real pandas code in a sandboxed environment to answer questions.

You upload a CSV. In under 30 seconds, she gives you:

- **A full statistical profile** — type inference, missingness, correlations, outliers
- **An AI-written report** — Executive Summary → Key Findings → Anomalies → Next Questions
- **Auto-generated charts** — correlation heatmap, scatter plots, distributions, category bars
- **A chat interface** where you can ask her anything about your data and see the pandas code she wrote to answer it

Every answer is grounded in your actual data. No made-up numbers. No fabricated columns. That's the whole point.

---

## ✨ Features

### 📥 Multi-format data ingestion
Reads CSV, TSV, Excel, Parquet, and JSON. Automatically detects column types (numeric, categorical, datetime, boolean, text, ID) and handles real-world messiness like duplicate column names and pandas' fake-datetime auto-parsing.

### 🔍 Automatic statistical profiling
Summary statistics, skewness, kurtosis, and cardinality per column. Detects strong correlations (|r| ≥ 0.3) across all numeric pairs. Flags outliers using two complementary methods: **IQR for per-column analysis** and **Isolation Forest for multivariate anomaly detection**.

### 📝 AI-written analysis reports
A senior-analyst-tone Markdown briefing streamed live from Groq's Llama 3.3 70B: Executive Summary → Data Quality → Key Findings → Anomalies & Outliers → Recommended Next Questions. Every number cited is traceable to the analysis context — no hallucination possible.

### 📈 Auto-generated Plotly visualizations
Chart selection is adaptive: correlation heatmap, scatter plots for strongly-correlated pairs, distribution histograms for the most variable numeric columns (ranked by coefficient of variation), bar charts for top categorical values, and time series when a datetime column exists.

### 💬 Q&A chat with sandboxed code execution
Ask questions in plain English. Vera writes pandas code, runs it in a **security-hardened sandbox**, and returns the answer plus a plain-English explanation. Every question shows the exact code that ran — full transparency.

### 🔒 AST-based security sandbox
No `exec(llm_output)` here. Every piece of generated code is parsed with Python's `ast` module and rejected if it contains imports, file I/O, `eval`, `exec`, `open`, or dunder-attribute sandbox escapes like `__class__.__subclasses__()`. Runs in a restricted namespace with only `df`, `pd`, `np`, and `plt` exposed.

---

## 🏗️ Architecture

![Vera.AI Architecture](docs/architecture.png)

Vera is composed of **four specialized AI agents** coordinated by a Streamlit orchestrator:

| Agent | Role | LLM? |
|:------|:-----|:----:|
| **Analyzer** | Orchestrates statistical analysis; produces an `AnalysisResult` shared with all downstream agents | No |
| **Report Agent** | Generates the streamed Markdown briefing from analysis context | Yes |
| **Chart Agent** | Auto-selects and renders Plotly visualizations based on column types | No |
| **Q&A Agent** | Writes pandas code → sandbox-executes → generates explanation (2 LLM calls per question) | Yes |

---

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Frontend</b></td>
<td>Streamlit · Custom CSS (Instrument Serif, Inter, JetBrains Mono)</td>
</tr>
<tr>
<td><b>Data</b></td>
<td>Pandas · NumPy · scikit-learn (Isolation Forest) · SciPy</td>
</tr>
<tr>
<td><b>Visualization</b></td>
<td>Plotly · Matplotlib · statsmodels (OLS trendlines)</td>
</tr>
<tr>
<td><b>LLM Orchestration</b></td>
<td>LangChain · Groq API · Llama 3.3 70B Versatile</td>
</tr>
<tr>
<td><b>Security</b></td>
<td>Python <code>ast</code> module · restricted-namespace <code>exec()</code></td>
</tr>
<tr>
<td><b>Deployment</b></td>
<td>Docker (Python 3.11-slim) · Hugging Face Spaces · GitHub</td>
</tr>
</table>

---

## 🚀 Quick Start

### Try it live
The fastest way — no install needed:
👉 **[huggingface.co/spaces/Paritosh271201/vera-ai](https://huggingface.co/spaces/Paritosh271201/vera-ai)**

### Run locally

```bash
# Clone
git clone https://github.com/ParitoshVyawahare/Vera.AI---A-End-to-END-Data-Analyst-agent.git
cd Vera.AI---A-End-to-END-Data-Analyst-agent

# Set up virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
cp .env.example .env
# Then edit .env and paste your key

# Run
streamlit run app.py
```

Vera opens at `http://localhost:8501`.

---

---

## 🧠 Design Decisions

A few small engineering choices that separate Vera from a tutorial clone:

- **Prompts live in their own module** (`prompts/templates.py`) — treat prompt engineering as code, not string literals scattered through agents.
- **The sandbox rejects, not sanitizes** — safer to fail-closed than try to clean up dangerous code. Rejected code returns a clear error to the user.
- **The Q&A agent uses two LLM calls** — one to write the pandas code, one to explain the result in plain English. This is why answers feel like they come from an analyst, not a query engine.
- **Multivariate + univariate outlier detection** — IQR alone finds per-column outliers; Isolation Forest finds *row-level* anomalies that no single column would flag. Both are shown separately.
- **Fake-datetime detection** — pandas' `read_csv` will happily interpret `08:57:13` as a datetime with today's date. Vera detects and undoes this to prevent misleading time-series plots.
- **CSS variables + Google Fonts loaded inline** — no build step, no bundler; ships as pure Streamlit but doesn't look like it.

---

## 🎨 Design Language

Vera's brand is built around **truth** (from Latin *veritas*). The visual identity reflects that:

| | |
|:---|:---|
| **Primary** | Deep evergreen `#0E5C4A` — trustworthy, not tech-bro blue |
| **Accent** | Warm terracotta `#D97757` — nods to the AI heritage |
| **Background** | Cream `#FBFAF7` — reduces glare, feels premium |
| **Typography** | Instrument Serif (headers) · Inter (body) · JetBrains Mono (numbers) |

The aesthetic is *modern data journal* — editorial, precise, calm.

---

## 🗺️ Roadmap

- [x] Multi-format data ingestion
- [x] Semantic column type inference
- [x] Statistical profiling + correlations + outliers
- [x] LLM-written report with streaming
- [x] Auto-generated Plotly visualizations
- [x] Q&A chat with sandboxed pandas execution
- [x] Plain-English answer explanations
- [x] Custom-branded UI
- [x] Live deployment on Hugging Face Spaces
- [ ] Multi-file support (join across CSVs)
- [ ] PDF export of the analysis report
- [ ] Suggested-question buttons from report insights
- [ ] Time-series forecasting when a real datetime column is present
- [ ] Semantic caching for repeat questions

---

## 📄 License

MIT — do what you want, but I'd love a shoutout if you build on it.

---

## 👤 Author

**Paritosh Vyawahare**

Built as a portfolio piece to demonstrate LLM agent design, safe code execution, and end-to-end product thinking — from data loading to deployment.

- 🌐 [Live demo](https://huggingface.co/spaces/Paritosh271201/vera-ai)
- 💼 [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-HANDLE)
- 🐙 [GitHub](https://github.com/ParitoshVyawahare)

---

<div align="center">

*Vera.AI — analysis you can cite.*

</div>
