"""Generates a written analysis report from AnalysisResult using Groq."""

from __future__ import annotations
import json
import os
from typing import Iterator

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from agents.analyzer import AnalysisResult
from prompts.templates import build_report_messages

load_dotenv()


def _get_llm(streaming: bool = True) -> ChatGroq:
    """Create the Groq LLM client. Reads key + model from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.2,
        streaming=streaming,
        max_tokens=2048,
    )


def _format_context(result: AnalysisResult, source_name: str) -> str:
    """Convert the analysis into a compact JSON block for the LLM.
    Smaller context = better reasoning + faster + cheaper."""
    ctx = result.to_context_dict()
    ctx["source_name"] = source_name
    return json.dumps(ctx, indent=2, default=str)


def _to_langchain_messages(msg_dicts: list[dict]):
    """Convert {role, content} dicts into LangChain message objects."""
    mapping = {"system": SystemMessage, "user": HumanMessage}
    return [mapping[m["role"]](content=m["content"]) for m in msg_dicts]


def generate_report(result: AnalysisResult, source_name: str) -> str:
    """Generate the full report as a single string (non-streaming)."""
    llm = _get_llm(streaming=False)
    context = _format_context(result, source_name)
    messages = _to_langchain_messages(build_report_messages(source_name, context))
    response = llm.invoke(messages)
    return response.content


def stream_report(result: AnalysisResult, source_name: str) -> Iterator[str]:
    """Yield report chunks as they arrive from Groq. Feeds Streamlit's write_stream."""
    llm = _get_llm(streaming=True)
    context = _format_context(result, source_name)
    messages = _to_langchain_messages(build_report_messages(source_name, context))
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content