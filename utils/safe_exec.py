"""Safely execute LLM-generated pandas code.

The sandbox does three things:
1. Parses the code with `ast` to reject dangerous constructs before running.
2. Runs the code in a restricted namespace (only pd, np, plt, and the
   provided dataframe(s) are exposed).
3. Captures the final expression's value as the "result" to return.

This is intentionally conservative — many things are blocked that could
theoretically be safe (like importing math), because a portfolio-grade tool
should fail closed, not open.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, no GUI popups
import matplotlib.pyplot as plt


# ─── Configuration ───────────────────────────────────────────────────
# Names the code is allowed to reference. Anything else → rejected.
ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
}

# AST node types that are always dangerous.
FORBIDDEN_NODES = (
    ast.Import,           # any import statement
    ast.ImportFrom,       # from x import y
    ast.Global,           # global variables
    ast.Nonlocal,
)

# Attribute access we refuse (used for sandbox escape attempts).
FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__builtins__", "__import__", "__code__",
    "__closure__", "__dict__", "__getattribute__",
}

# Function names we refuse regardless of context.
FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "help", "memoryview",
}


@dataclass
class ExecutionResult:
    """Outcome of a sandboxed run."""
    success: bool
    value: Any = None                # last expression's value (table, scalar, etc.)
    figure: Any = None               # matplotlib figure if code created one
    stdout: str = ""                 # captured print() output
    error: str | None = None         # error message if success=False
    code: str = ""                   # the code that was run (for display)


# ─── Static safety check ─────────────────────────────────────────────
def _check_ast_safety(code: str) -> str | None:
    """Return None if safe, or an error message describing why not."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e.msg}"

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return f"Disallowed construct: {type(node).__name__}"

        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return f"Disallowed attribute access: .{node.attr}"

        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALLS:
                return f"Disallowed name: {node.id}"

        if isinstance(node, ast.Call):
            # e.g. someone calling __builtins__["eval"]
            if isinstance(node.func, ast.Subscript):
                return "Disallowed dynamic call"
    return None


# ─── Execution ───────────────────────────────────────────────────────
def _split_last_expr(code: str) -> tuple[str, str | None]:
    """Split code into (statements, last_expression_if_any).
    So we can eval() the final line and capture its result — like a notebook."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None

    if not tree.body:
        return code, None

    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        body_stmts = tree.body[:-1]
        body_code = ast.unparse(ast.Module(body=body_stmts, type_ignores=[]))
        last_expr = ast.unparse(last.value)
        return body_code, last_expr
    return code, None


def safe_execute(
    code: str,
    df: pd.DataFrame | None = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
) -> ExecutionResult:
    """Run LLM-generated code against one or more dataframes.

    Two modes:
    - Single-file (backward-compat): pass `df=some_df`. It's exposed
      in the sandbox as the variable `df`.
    - Multi-file: pass `dataframes={"df_sales": s, "df_products": p}`.
      Each entry is exposed under its dict key, so the LLM can reference
      them by name (e.g. `df_sales.merge(df_products, on='id')`).

    You can also pass both — `df` becomes the variable `df`, and each
    entry in `dataframes` is exposed under its own name. All dataframes
    are copied to prevent the sandbox from mutating the caller's data.
    """
    # Static safety check
    reason = _check_ast_safety(code)
    if reason is not None:
        return ExecutionResult(success=False, error=reason, code=code)

    # Build the restricted builtins
    safe_builtins = {name: __builtins__[name] if isinstance(__builtins__, dict)
                     else getattr(__builtins__, name)
                     for name in ALLOWED_BUILTINS
                     if (name in __builtins__ if isinstance(__builtins__, dict)
                         else hasattr(__builtins__, name))}

    ns: dict[str, Any] = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
    }

    # Expose dataframes — single `df`, a dict of named ones, or both
    if df is not None:
        ns["df"] = df.copy()
    if dataframes:
        for name, frame in dataframes.items():
            ns[name] = frame.copy()

    # Capture stdout
    import io, contextlib
    buf = io.StringIO()

    # Split off the last expression to capture its value
    body_code, last_expr = _split_last_expr(code)

    plt.close("all")   # clean slate for any figures the code creates

    try:
        with contextlib.redirect_stdout(buf):
            if body_code.strip():
                exec(compile(body_code, "<sandbox>", "exec"), ns)
            value = None
            if last_expr:
                value = eval(compile(last_expr, "<sandbox>", "eval"), ns)
    except Exception as e:
        return ExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            stdout=buf.getvalue(),
            code=code,
        )

    # Capture any figure the code left open
    fig = plt.gcf() if plt.get_fignums() else None

    return ExecutionResult(
        success=True,
        value=value,
        figure=fig,
        stdout=buf.getvalue(),
        code=code,
    )