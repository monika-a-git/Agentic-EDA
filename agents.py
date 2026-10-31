import builtins
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from openai import OpenAI

MAX_RETRIES = 2

# Tried in order. First one that works wins. Check each provider's docs if a model name changes.
PROVIDERS = [
    {
        "name": "Groq",
        "secret": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile"],
    },
    {
        "name": "Gemini",
        "secret": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
    },
    {
        "name": "OpenAI",
        "secret": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
]


def get_api_key(secret_name: str) -> str:
    """Retrieve an API key from st.session_state, st.secrets, or environment variables."""
    # 1. Streamlit session state (user entered via UI)
    try:
        if secret_name in st.session_state and st.session_state[secret_name]:
            val = str(st.session_state[secret_name]).strip()
            if val and not val.startswith("your_") and not val.startswith("gsk_your_"):
                return val
    except Exception:
        pass

    # 2. Streamlit secrets
    try:
        if secret_name in st.secrets:
            val = str(st.secrets[secret_name]).strip()
            if val and not val.startswith("your_") and not val.startswith("gsk_your_"):
                return val
    except Exception:
        pass

    # 3. Environment variables
    val = os.environ.get(secret_name, "").strip()
    if val and not val.startswith("your_") and not val.startswith("gsk_your_"):
        return val

    return ""


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    errors = []
    for p in PROVIDERS:
        key = get_api_key(p["secret"])
        if not key:
            continue

        base_url = p["base_url"]
        models = p.get("models", [p.get("model")])

        # If an OpenAI key (sk-...) was mistakenly placed in GEMINI_API_KEY, route to OpenAI
        if p["name"] == "Gemini" and key.startswith("sk-"):
            base_url = "https://api.openai.com/v1"
            models = ["gpt-4o-mini", "gpt-4o"]

        client = OpenAI(api_key=key, base_url=base_url, timeout=30, max_retries=0)

        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=1500,
                )
                try:
                    st.session_state["last_provider"] = f"{p['name']} ({model})"
                except Exception:
                    pass
                return response.choices[0].message.content
            except Exception as e:
                errors.append(f"{p['name']}/{model}: {type(e).__name__}")
                continue

    if not errors:
        raise RuntimeError("No API keys found. Please provide a GROQ_API_KEY or GEMINI_API_KEY in secrets.toml or the sidebar.")
    raise RuntimeError("All providers failed -> " + "; ".join(errors))


CODER_SYSTEM = """You are a senior data scientist who writes clean, safe Python for EDA.

Rules:
- A pandas DataFrame named `df` is ALREADY loaded. Never read any files.
- `pd`, `np`, `plt` and `sns` are ALREADY imported. Do NOT write import statements.
- Use ONLY column names that appear in the metadata provided.
- Create each chart with its own plt.figure(figsize=(8, 4)) and give it a title.
- Do NOT call plt.show() or plt.savefig().
- Never use os, sys, subprocess, open(), eval(), exec() or network access.
- If a step is impossible for this data, skip it.
- Return ONLY executable Python code. No explanations, no markdown fences."""

ANALYST_SYSTEM = """You are a senior data scientist explaining findings to a business stakeholder.
Base every statement ONLY on the statistics provided.
If something cannot be determined from them, say so. Never invent numbers."""

VERIFIER_SYSTEM = """You turn statements from a data report into machine-checkable tests.

Return ONLY a valid JSON list (no markdown fences, no commentary). Each item must be:
{"claim": "<the statement>", "claimed_value": <number>, "code": "<pandas code>"}

Rules for `code`:
- A DataFrame `df` is already loaded; `pd` and `np` are imported. No import statements.
- It must store ONE single number in a variable named `result`.
- Use ONLY column names from the metadata.
Rules for choosing claims:
- Only include claims that state a specific number about the data (mean, max, min, count,
  percentage missing, correlation, and so on). Skip vague or opinion statements.
- Maximum 8 claims."""


def clean_code(raw: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)(?:```|$)", raw, re.DOTALL)
    code = match.group(1) if match else raw
    lines = [l for l in code.splitlines() if not re.match(r"\s*(import|from)\s+\w", l)]
    return "\n".join(lines).replace("plt.show()", "").strip()


BLOCKED_PATTERNS = [
    r"\b(os|sys|subprocess|shutil|socket|requests|pathlib)\b",
    r"\b(open|eval|exec|compile|input|globals|locals|getattr|setattr)\s*\(",
    r"\.(to_csv|to_pickle|to_excel|to_parquet|to_sql|read_\w+)\s*\(",
    r"__",
]


def is_safe(code: str):
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return False, f"Blocked unsafe pattern: {pattern}"
    return True, None


ALLOWED_BUILTINS = [
    "len", "range", "min", "max", "sum", "abs", "round", "sorted", "enumerate",
    "zip", "list", "dict", "set", "tuple", "float", "int", "str", "bool",
    "print", "isinstance", "any", "all", "map", "filter", "reversed",
]
SAFE_BUILTINS = {name: getattr(builtins, name) for name in ALLOWED_BUILTINS}


def run_code(code: str, df: pd.DataFrame):
    """Returns (figures, error_message, environment)."""
    ok, reason = is_safe(code)
    if not ok:
        return [], reason, {}
    plt.close("all")
    env = {"__builtins__": SAFE_BUILTINS, "df": df.copy(),
           "pd": pd, "np": np, "plt": plt, "sns": sns}
    try:
        exec(code, env)
    except Exception as e:
        plt.close("all")
        return [], f"{type(e).__name__}: {e}", {}
    return [plt.figure(n) for n in plt.get_fignums()], None, env


def coder_agent(schema_text: str, df: pd.DataFrame) -> dict:
    task = f"""Dataset metadata:
{schema_text}

Write Python code to:
1. Plot histograms for up to 6 numeric columns
2. Plot a correlation heatmap of the numeric columns
3. Plot a bar chart of missing values per column (skip if none are missing)
4. If a date column exists, plot the first numeric column over time"""

    code = clean_code(call_llm(CODER_SYSTEM, task))
    error = None
    for attempt in range(MAX_RETRIES + 1):
        figs, error, _ = run_code(code, df)
        if error is None:
            return {"code": code, "figs": figs, "retries": attempt, "error": None}
        if attempt < MAX_RETRIES:
            fix = f"""Your code failed.

Metadata:
{schema_text}

Code:
{code}

Error:
{error}

Return the full corrected code only."""
            code = clean_code(call_llm(CODER_SYSTEM, fix))
    return {"code": code, "figs": [], "retries": MAX_RETRIES, "error": error}


def chat_agent(question: str, schema_text: str, df: pd.DataFrame) -> dict:
    task = f"""Dataset metadata:
{schema_text}

User question: {question}

Write Python code that answers the question.
Store the final result in a variable named `answer` (a number, string or DataFrame).
Optionally create ONE chart if it helps."""

    code = clean_code(call_llm(CODER_SYSTEM, task))
    error = None
    for attempt in range(MAX_RETRIES + 1):
        figs, error, env = run_code(code, df)
        if error is None:
            return {"code": code, "figs": figs, "answer": env.get("answer"), "error": None}
        if attempt < MAX_RETRIES:
            fix = f"Question: {question}\nCode:\n{code}\nError:\n{error}\nReturn corrected code only."
            code = clean_code(call_llm(CODER_SYSTEM, fix))
    return {"code": code, "figs": [], "answer": None, "error": error}


def analyst_agent(schema_text: str, summary: str, correlations: str, issues: str = "") -> str:
    task = f"""Metadata:
{schema_text}

Summary statistics (one row per column):
{summary}

Strongest numeric correlations (computed with Pandas):
{correlations}

Detected data quality issues:
{issues or "None detected"}

Write a short report in markdown with these sections:
## Key patterns
## Data quality issues
## Outliers and anomalies
## Interesting relationships
## Modeling recommendations"""
    return call_llm(ANALYST_SYSTEM, task, temperature=0.4)


def extract_claims(report: str, schema_text: str) -> list:
    raw = call_llm(VERIFIER_SYSTEM, f"Metadata:\n{schema_text}\n\nReport:\n{report}", temperature=0.0)
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            claims = json.loads(raw[start:end + 1])
            return claims if isinstance(claims, list) else []
        except ValueError:  # includes JSONDecodeError
            pass
    return []


def verify_claims(report: str, schema_text: str, df: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    rows = []
    for item in extract_claims(report, schema_text)[:8]:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", ""))
        code = clean_code(str(item.get("code", "")))

        try:
            claimed = float(item.get("claimed_value"))
        except (TypeError, ValueError):
            claimed = None

        _, error, env = run_code(code, df)
        actual = None
        if error is None:
            try:
                res_val = env.get("result")
                if isinstance(res_val, (pd.Series, np.ndarray)) and len(res_val) == 1:
                    actual = float(res_val.iloc[0] if isinstance(res_val, pd.Series) else res_val[0])
                else:
                    actual = float(res_val)
            except (TypeError, ValueError):
                actual = None

        if claimed is None or actual is None or not math.isfinite(actual):
            status = "⚠️ Could not check"
        else:
            diff = abs(actual - claimed) / max(abs(actual), 1e-9)
            status = "✅ Verified" if diff <= tol else "❌ Mismatch"

        rows.append({
            "claim": claim,
            "claimed": claimed,
            "actual": None if actual is None else round(actual, 4),
            "status": status,
            "code": code,
        })
    return pd.DataFrame(rows, columns=["claim", "claimed", "actual", "status", "code"])


def explain_ml(task: str, target: str, results: pd.DataFrame, importance: pd.Series, dropped: list) -> str:
    prompt = f"""A baseline ML experiment predicting `{target}` ({task}).

Model comparison (test set):
{results.to_string(index=False)}

Top feature importances (Random Forest):
{importance.round(4).to_string()}

Columns excluded from training: {dropped or "none"}

In 5-7 sentences:
- Say whether the models beat the dummy baseline, and by how much.
- Name the most influential features.
- If any model scores suspiciously high (accuracy or R2 above 0.98), warn about possible data leakage
  (for example, predicting a price from other columns of the same day).
- Suggest one concrete next step.
Use ONLY the numbers above."""
    return call_llm(ANALYST_SYSTEM, prompt, temperature=0.3)
