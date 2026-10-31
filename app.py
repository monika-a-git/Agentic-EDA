import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agents import analyst_agent, chat_agent, coder_agent, explain_ml, verify_claims
from eda_tools import build_summary, extract_schema, health_report, top_correlations
from ml_baseline import run_baseline

st.set_page_config(
    page_title="Agentic EDA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Retro-Chic Editorial CSS (Reference Design) ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Background Canvas: Warm cream with subtle grid */
.stApp {
    background-color: #FAF8F5;
    background-image: radial-gradient(#e5e2da 1px, transparent 1px);
    background-size: 28px 28px;
}

/* Serif typography */
.serif-title {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
    color: #18181B;
}

/* ---------------- SIDEBAR (Matches Whole Web UI Theme) ---------------- */
[data-testid="stSidebar"] {
    background-color: #F5F2EB !important;
    border-right: 1.5px solid #18181B !important;
}

[data-testid="stSidebar"] hr {
    border-color: #E2DDD5 !important;
}

.sidebar-brand-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #18181B;
    letter-spacing: -0.02em;
    padding: 0.5rem 0 1rem 0;
}

/* Transform st.radio into clean navigation bar pills matching theme */
[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
    gap: 6px;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: transparent;
    padding: 9px 16px;
    border-radius: 9999px;
    color: #52525B;
    font-size: 0.92rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1.5px solid transparent;
    margin: 0;
    width: 100%;
    display: flex;
    align-items: center;
}

/* Hide default radio circles */
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color: #18181B;
    background: #FFFFFF;
    border: 1.5px solid #18181B;
    box-shadow: 2px 2px 0px #18181B;
}

[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #18181B !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border: 1.5px solid #18181B !important;
    box-shadow: 2px 2px 0px #E11D48 !important;
}

/* ---------------- HERO HEADER (Fixed alignment, no link icons) ---------------- */
.hero-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 1.5rem 1rem 2rem 1rem;
    max-width: 900px;
    margin: 0 auto;
}

.hero-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFE4E8;
    color: #9F1239;
    border: 1.5px solid #FDA4AF;
    padding: 5px 16px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.hero-heading {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(2.4rem, 4.8vw, 3.8rem);
    line-height: 1.15;
    color: #18181B;
    margin: 0 0 1rem 0;
    text-align: center;
}

.hero-highlight {
    background: #FECDD3;
    padding: 2px 14px;
    border-radius: 10px;
    border: 1.5px solid #18181B;
    display: inline-block;
    color: #18181B;
    margin-left: 6px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #52525B;
    max-width: 680px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.6;
    text-align: center;
}

/* ---------------- TICKER RIBBON ---------------- */
.ticker-ribbon {
    background: #D4ED82;
    color: #1A3800;
    border-top: 1.5px solid #18181B;
    border-bottom: 1.5px solid #18181B;
    padding: 10px 0;
    margin: 1.2rem -3rem 2.2rem -3rem;
    text-align: center;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    overflow: hidden;
    white-space: nowrap;
}

/* ---------------- 4 FEATURE CARDS ---------------- */
.feature-card {
    border: 1.5px solid #18181B;
    border-radius: 18px;
    padding: 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 4px 4px 0px #18181B;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    background: #FFFFFF;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 6px 6px 0px #18181B;
}

.card-rose { background-color: #FFF1F2; }
.card-sky { background-color: #F0F9FF; }
.card-mint { background-color: #F0FDF4; }
.card-amber { background-color: #FEFCE8; }

.card-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    border: 1px solid #18181B;
    background: #FFFFFF;
}

.card-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.35rem;
    color: #18181B;
    margin-bottom: 6px;
}

.card-desc {
    font-size: 0.85rem;
    color: #52525B;
    line-height: 1.5;
}

/* ---------------- STUDIO BOX CONTAINERS ---------------- */
.studio-box {
    background: #FFFFFF;
    border: 1.5px solid #18181B;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 4px 4px 0px #18181B;
    margin-bottom: 1.8rem;
}

.studio-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1.5px solid #E4E4E7;
    padding-bottom: 12px;
    margin-bottom: 16px;
}

.studio-title {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.6rem;
    color: #18181B;
    margin: 0;
}

/* ---------------- BUTTONS ---------------- */
.stButton > button {
    border-radius: 9999px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.6rem !important;
    border: 1.5px solid #18181B !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"] {
    background: #18181B !important;
    color: #FFFFFF !important;
    box-shadow: 3px 3px 0px #E11D48 !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 5px 5px 0px #E11D48 !important;
}

.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #18181B !important;
    box-shadow: 3px 3px 0px #18181B !important;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Sidebar (Matches Whole Web UI Theme - Private Keys) ----------
with st.sidebar:
    st.markdown(
        """
    <div class="sidebar-brand-title">
        Agentic <span style="background:#FECDD3; border:1.5px solid #18181B; padding:1px 8px; border-radius:8px; font-size:1.15rem; font-family:'Plus Jakarta Sans', sans-serif; font-weight:800;">EDA</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    nav_selection = st.radio(
        "Navigation",
        ["Overview", "Charts", "Insights", "Verification", "ML baseline", "Ask your data"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Private status badge (matching warm theme, no keys displayed)
    st.markdown(
        """
    <div style="background:#F0FDF4; border:1.5px solid #18181B; border-radius:12px; padding:12px; margin-bottom:12px; box-shadow:3px 3px 0px #18181B;">
        <div style="display:flex; align-items:center; gap:8px; font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#15803D;">
            <span style="height:8px; width:8px; background:#22C55E; border:1px solid #14532D; border-radius:50%; display:inline-block;"></span>
            Multi-Model Fallback
        </div>
        <div style="font-size:0.78rem; color:#374151; margin-top:4px; font-weight:600;">
            Groq ➡️ Gemini 2.5 Flash
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    last_p = st.session_state.get("last_provider")
    if last_p:
        st.markdown(
            f"""
        <div style="background:#FFFFFF; border:1.5px solid #18181B; border-radius:10px; padding:8px 12px; font-size:0.76rem; color:#18181B; margin-bottom:12px; box-shadow:2px 2px 0px #18181B;">
            <span style="color:#71717A; font-weight:600;">Last used:</span> <strong>{last_p}</strong>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if st.button("🧹 Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# State persistence
df = st.session_state.get("df")

# ==============================================================================
# 1. OVERVIEW PAGE
# ==============================================================================
if nav_selection == "Overview":
    # Hero Section with aligned text & sticker badge (no anchor links)
    st.markdown(
        """
    <div class="hero-wrapper">
        <div class="hero-pill-badge">
            <span>🍒</span> WHERE RAW DATA MEETS AUTONOMOUS AI
        </div>
        <div class="hero-heading">
            The Multi-agent Pipeline<br/>
            For <span class="hero-highlight">Agentic EDA</span>
        </div>
        <div class="hero-subtitle">
            Upload your tabular dataset. One AI agent writes and repairs visualization code,
            another translates numbers into strategic insights, and a third fact-checks every claim.
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Matcha Green Ticker Ribbon
    st.markdown(
        """
    <div class="ticker-ribbon">
        ✦ CODER AGENT (SANDBOXED PYTHON) &nbsp;&nbsp;&nbsp;&nbsp; ✦ BUSINESS ANALYST &nbsp;&nbsp;&nbsp;&nbsp; ✦ FACT-CHECKED AUDITOR &nbsp;&nbsp;&nbsp;&nbsp; ✦ AUTOMATED ML BASELINES &nbsp;&nbsp;&nbsp;&nbsp; ✦ LIVE DATA CHAT ✦
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 4 Feature Showcase Cards (matching the screenshot reference)
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.markdown(
            """
        <div class="feature-card card-rose">
            <div>
                <span class="card-tag" style="background:#FFE4E8; color:#9F1239;">Agent 01</span>
                <div class="card-title">Coder Agent</div>
                <p class="card-desc">Writes sandboxed Matplotlib and Seaborn plotting code with automated self-healing error recovery.</p>
            </div>
            <div style="margin-top:14px; font-size:0.75rem; font-weight:700; color:#9F1239;">📈 4 Native Plots</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_c2:
        st.markdown(
            """
        <div class="feature-card card-sky">
            <div>
                <span class="card-tag" style="background:#E0F2FE; color:#0369A1;">Agent 02</span>
                <div class="card-title">Analyst Agent</div>
                <p class="card-desc">Synthesizes descriptive statistics, correlations, and anomalies into an executive-ready business report.</p>
            </div>
            <div style="margin-top:14px; font-size:0.75rem; font-weight:700; color:#0369A1;">🧠 Strategic Insights</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_c3:
        st.markdown(
            """
        <div class="feature-card card-mint">
            <div>
                <span class="card-tag" style="background:#DCFCE7; color:#15803D;">Agent 03</span>
                <div class="card-title">Verifier Agent</div>
                <p class="card-desc">Pulls numeric statements from the report and audits each figure against the real dataset with code proofs.</p>
            </div>
            <div style="margin-top:14px; font-size:0.75rem; font-weight:700; color:#15803D;">✅ 0% Hallucinations</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_c4:
        st.markdown(
            """
        <div class="feature-card card-amber">
            <div>
                <span class="card-tag" style="background:#FEF9C3; color:#854D0E;">Agent 04</span>
                <div class="card-title">ML & Live Chat</div>
                <p class="card-desc">Trains instant classification or regression baseline pipelines, plus lets you talk directly with your dataset.</p>
            </div>
            <div style="margin-top:14px; font-size:0.75rem; font-weight:700; color:#854D0E;">🤖 Predictive Studio</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # Step 1: Upload CSV
    st.markdown(
        """
    <div class="studio-box">
        <div class="studio-header">
            <h3 class="studio-title">📁 Step 1: Upload Dataset</h3>
            <span style="font-size:0.8rem; font-weight:800; background:#F4F4F5; border:1px solid #18181B; padding:4px 12px; border-radius:9999px;">CSV Format</span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"], label_visibility="collapsed", key="csv_uploader_overview"
    )

    if uploaded_file is not None:
        if st.session_state.get("file_name") != uploaded_file.name:
            st.session_state["file_name"] = uploaded_file.name
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state.pop("result", None)
                st.session_state.pop("ml", None)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                df = None

    st.markdown("</div>", unsafe_allow_html=True)

    if df is None:
        st.markdown(
            """
        <div style="text-align:center; padding:3rem 1rem; color:#71717A;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📂</div>
            <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.4rem; color:#18181B; margin-bottom:0.4rem;">
                No dataset loaded yet
            </div>
            <p style="font-size:0.9rem;">Upload a CSV file above to begin your agentic data science pipeline.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.stop()

    # Step 2: Data Preview & Health Audit
    health = health_report(df)
    st.markdown(
        f"""
    <div class="studio-box">
        <div class="studio-header">
            <h3 class="studio-title">🔍 Step 2: Data Preview & Quality Audit</h3>
            <span style="font-size:0.8rem; font-weight:800; background:#E0F2FE; border:1.5px solid #18181B; padding:4px 12px; border-radius:9999px;">
                {df.shape[0]:,} Rows × {df.shape[1]} Columns
            </span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col_score, col_issues = st.columns([1, 2.5])
    with col_score:
        score = health["score"]
        score_color = "#15803D" if score >= 80 else "#B45309" if score >= 60 else "#B91C1C"
        score_bg = "#DCFCE7" if score >= 80 else "#FEF3C7" if score >= 60 else "#FEE2E2"

        st.markdown(
            f"""
        <div style="background:{score_bg}; border:1.5px solid #18181B; border-radius:14px; padding:18px; text-align:center; box-shadow:3px 3px 0px #18181B;">
            <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:{score_color};">Health Score</div>
            <div style="font-family:'DM Serif Display', Georgia, serif; font-size:3rem; color:#18181B; line-height:1; margin:8px 0;">
                {score}<span style="font-size:1.2rem; color:#71717A;">/100</span>
            </div>
            <div style="font-size:0.8rem; font-weight:700; color:{score_color};">
                {"✨ High Quality" if score >= 80 else "⚠️ Noticeable Issues" if score >= 60 else "🚨 Major Issues"}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_issues:
        st.markdown(
            "<div style='font-size:0.85rem; font-weight:700; color:#18181B; margin-bottom:8px;'>Data Quality Diagnoses:</div>",
            unsafe_allow_html=True,
        )
        if health["issues"]:
            for issue in health["issues"]:
                st.markdown(
                    f"<div style='background:#FFFBEB; border:1px solid #FCD34D; border-radius:8px; padding:6px 12px; font-size:0.82rem; margin-bottom:6px; color:#92400E;'>⚠️ {issue}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='background:#F0FDF4; border:1px solid #86EFAC; border-radius:8px; padding:8px 12px; font-size:0.82rem; color:#166534;'>✅ Clean dataset: no major missing rates, duplicates, or constant columns detected.</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("")
    btn_col1, btn_col2 = st.columns([1.6, 3])
    with btn_col1:
        run_eda_btn = st.button("🚀 Run Agentic EDA Pipeline", type="primary", use_container_width=True)
    with btn_col2:
        st.caption("Dispatches Coder Agent, Analyst Agent, and Verifier Agent simultaneously.")

    st.markdown("</div>", unsafe_allow_html=True)

    if run_eda_btn:
        try:
            schema_text = extract_schema(df)
            summary = build_summary(df)
            correlations = top_correlations(df)

            with st.status("🤖 Collaborative Agents at Work...", expanded=True) as status:
                st.write("1️⃣ **Coder Agent**: Writing and executing safe plotting code...")
                coder = coder_agent(schema_text, df)

                st.write("2️⃣ **Analyst Agent**: Synthesizing statistics & anomaly report...")
                insights = analyst_agent(schema_text, summary, correlations, "; ".join(health["issues"]))

                st.write("3️⃣ **Verifier Agent**: Auditing numeric claims against dataset rows...")
                verification = verify_claims(insights, schema_text, df)

                status.update(label="✅ Agentic EDA Pipeline Complete!", state="complete", expanded=False)

            st.session_state["result"] = {
                "coder": coder,
                "insights": insights,
                "verification": verification,
            }
        except Exception as e:
            st.error(f"Error running pipeline: {e}")

    result = st.session_state.get("result")
    if result:
        st.markdown(
            """
        <div style="margin:2rem 0 1rem 0;">
            <div class="hero-pill-badge" style="background:#DCFCE7; border-color:#86EFAC; color:#15803D;">
                <span>✨</span> Pipeline Results Ready
            </div>
            <h2 style="font-family:'DM Serif Display', Georgia, serif; font-size:2rem; color:#18181B; margin:0 0 0.4rem 0;">
                Results Overview
            </h2>
            <p style="font-size:0.9rem; color:#52525B;">
                Use the left navigation bar to explore the complete Charts, Insights, Verification audit, ML baseline, and Data Chat.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        coder = result["coder"]
        v = result["verification"]
        checked = v[v["status"] != "⚠️ Could not check"] if not v.empty else pd.DataFrame()
        ok_count = int((checked["status"] == "✅ Verified").sum()) if not checked.empty else 0
        total_claims = len(checked) if not checked.empty else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
            <div class="feature-card card-rose" style="padding:14px; box-shadow:3px 3px 0px #18181B;">
                <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#9F1239;">Visuals</div>
                <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.8rem; color:#18181B; margin:4px 0;">
                    {len(coder.get("figs", []))} Charts
                </div>
                <div style="font-size:0.75rem; color:#881337;">Explore in Charts tab</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                """
            <div class="feature-card card-sky" style="padding:14px; box-shadow:3px 3px 0px #18181B;">
                <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#0369A1;">Report</div>
                <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.8rem; color:#18181B; margin:4px 0;">
                    Complete
                </div>
                <div style="font-size:0.75rem; color:#075985;">Explore in Insights tab</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
            <div class="feature-card card-mint" style="padding:14px; box-shadow:3px 3px 0px #18181B;">
                <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#15803D;">Fact-Checked</div>
                <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.8rem; color:#18181B; margin:4px 0;">
                    {ok_count} / {total_claims}
                </div>
                <div style="font-size:0.75rem; color:#166534;">Explore in Verification tab</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with k4:
            st.markdown(
                """
            <div class="feature-card card-amber" style="padding:14px; box-shadow:3px 3px 0px #18181B;">
                <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#854D0E;">Predictive Lab</div>
                <div style="font-family:'DM Serif Display', Georgia, serif; font-size:1.8rem; color:#18181B; margin:4px 0;">
                    Unlocked
                </div>
                <div style="font-size:0.75rem; color:#713F12;">Explore in ML Baseline tab</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# 2. CHARTS PAGE
# ==============================================================================
elif nav_selection == "Charts":
    st.markdown(
        """
    <div style="margin-bottom:1.5rem;">
        <span class="card-tag" style="background:#FFE4E8; color:#9F1239;">Agent 01</span>
        <h2 class="serif-title" style="font-size:2.2rem; margin:6px 0;">Visual Analytics Gallery</h2>
        <p style="font-size:0.9rem; color:#52525B;">Histograms, correlation heatmap, missingness, and trend curves generated safely.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Please upload a CSV file in Overview first.")
        st.stop()

    result = st.session_state.get("result")
    if not result:
        st.info("Run the Agentic EDA pipeline in the **Overview** tab to generate charts.")
        st.stop()

    coder = result["coder"]
    if coder.get("error"):
        st.warning(f"Coder agent error: {coder['error']}")
    else:
        figs = coder.get("figs", [])
        if not figs:
            st.info("No charts generated.")
        else:
            if coder.get("retries"):
                st.caption(f"Self-healed code after {coder['retries']} retry(ies).")

            g1, g2 = st.columns(2)
            for i, fig in enumerate(figs):
                col = g1 if i % 2 == 0 else g2
                with col:
                    st.markdown(
                        f"""
                    <div style="background:#FFFFFF; border:1.5px solid #18181B; border-radius:16px; padding:16px; margin-bottom:18px; box-shadow:3px 3px 0px #18181B;">
                        <div style="font-size:0.75rem; font-weight:800; color:#71717A; margin-bottom:6px;">Chart {i+1}</div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
    <div class="studio-box">
        <div style="font-size:0.9rem; font-weight:800; margin-bottom:8px;">Safe Execution Sandbox Code:</div>
    """,
        unsafe_allow_html=True,
    )
    st.code(coder.get("code", "# No code available"), language="python")
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 3. INSIGHTS PAGE
# ==============================================================================
elif nav_selection == "Insights":
    st.markdown(
        """
    <div style="margin-bottom:1.5rem;">
        <span class="card-tag" style="background:#E0F2FE; color:#0369A1;">Agent 02</span>
        <h2 class="serif-title" style="font-size:2.2rem; margin:6px 0;">Analyst Agent Report</h2>
        <p style="font-size:0.9rem; color:#52525B;">Executive business interpretation of patterns, distributions, and anomalies.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Please upload a CSV file in Overview first.")
        st.stop()

    result = st.session_state.get("result")
    if not result:
        st.info("Run the Agentic EDA pipeline in the **Overview** tab to generate insights.")
        st.stop()

    insights = result["insights"]

    st.markdown(
        """
    <div class="studio-box">
    """,
        unsafe_allow_html=True,
    )
    st.markdown(insights)
    st.write("")
    st.download_button(
        "📥 Download Executive Report (.md)",
        data=insights,
        file_name=f"EDA_Report_{st.session_state.get('file_name', 'data')}.md",
        mime="text/markdown",
        type="primary",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 4. VERIFICATION PAGE
# ==============================================================================
elif nav_selection == "Verification":
    st.markdown(
        """
    <div style="margin-bottom:1.5rem;">
        <span class="card-tag" style="background:#DCFCE7; color:#15803D;">Agent 03</span>
        <h2 class="serif-title" style="font-size:2.2rem; margin:6px 0;">Verified Insights Fact-Checker</h2>
        <p style="font-size:0.9rem; color:#52525B;">Audits every numeric claim from the report directly against actual dataset values.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Please upload a CSV file in Overview first.")
        st.stop()

    result = st.session_state.get("result")
    if not result:
        st.info("Run the Agentic EDA pipeline in the **Overview** tab to view verified claims.")
        st.stop()

    v = result["verification"]
    st.markdown(
        """
    <div class="studio-box">
    """,
        unsafe_allow_html=True,
    )

    if v.empty:
        st.info("No checkable numeric claims were detected in the report.")
    else:
        checked = v[v["status"] != "⚠️ Could not check"]
        ok = int((checked["status"] == "✅ Verified").sum())
        total = len(checked)

        st.metric("Claims Verified Against Ground Truth", f"{ok} / {total}")
        st.write("")

        st.dataframe(
            v[["claim", "claimed", "actual", "status"]],
            use_container_width=True,
            column_config={
                "claim": st.column_config.TextColumn("Report Claim", width="large"),
                "claimed": st.column_config.NumberColumn("Claimed Value"),
                "actual": st.column_config.NumberColumn("Actual Computed Value"),
                "status": st.column_config.TextColumn("Audit Status"),
            },
        )

        with st.expander("🔬 View Verification Assertion Proofs"):
            for idx, row in v.iterrows():
                st.markdown(f"**Claim #{idx+1}:** {row['claim']}")
                st.code(row["code"], language="python")
                st.write("---")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 5. ML BASELINE PAGE
# ==============================================================================
elif nav_selection == "ML baseline":
    st.markdown(
        """
    <div style="margin-bottom:1.5rem;">
        <span class="card-tag" style="background:#FEF9C3; color:#854D0E;">Agent 04</span>
        <h2 class="serif-title" style="font-size:2.2rem; margin:6px 0;">Automated ML Baseline</h2>
        <p style="font-size:0.9rem; color:#52525B;">Automatic classification or regression benchmarking vs. Dummy and Random Forest.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Please upload a CSV file in Overview first.")
        st.stop()

    st.markdown(
        """
    <div class="studio-box">
        <div class="studio-header">
            <h3 class="studio-title">Target & Configuration</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        target = st.selectbox("Select Target Column to Predict", df.columns, key="ml_target_select")
    with c2:
        st.write("")
        st.write("")
        time_ordered = st.checkbox("Time-ordered split (no shuffle)", key="ml_time_ordered")

    st.write("")
    if st.button("⚡ Train Baseline Models", type="primary"):
        try:
            with st.spinner("Preparing features and training baseline models..."):
                ml = run_baseline(df, target, time_ordered)
                ml["explanation"] = explain_ml(
                    ml["task"], target, ml["results"], ml["importance"], ml["dropped"]
                )
            st.session_state["ml"] = ml
        except Exception as e:
            st.error(f"Training failed: {e}")

    ml = st.session_state.get("ml")
    if ml:
        st.markdown(
            f"""
        <div style="background:#F0FDF4; border:1px solid #86EFAC; border-radius:10px; padding:10px 14px; margin:14px 0; font-size:0.85rem; color:#166534;">
            <strong>Task Detected:</strong> {ml['task'].upper()} &nbsp;|&nbsp;
            <strong>Train Rows:</strong> {ml['n_train']:,} &nbsp;|&nbsp;
            <strong>Test Rows:</strong> {ml['n_test']:,}
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.dataframe(ml["results"], use_container_width=True)

        st.markdown(
            "<div style='font-size:0.9rem; font-weight:800; margin:16px 0 8px 0;'>Top Feature Importances (Random Forest):</div>",
            unsafe_allow_html=True,
        )
        st.bar_chart(ml["importance"])

        st.markdown(
            """
        <div style="background:#F8FAFC; border:1.5px solid #18181B; border-radius:12px; padding:16px; margin-top:14px; font-size:0.88rem; line-height:1.6;">
        """,
            unsafe_allow_html=True,
        )
        st.markdown(ml["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)

        if ml.get("dropped"):
            st.caption(f"Columns automatically excluded: {ml['dropped']}")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 6. ASK YOUR DATA PAGE
# ==============================================================================
elif nav_selection == "Ask your data":
    st.markdown(
        """
    <div style="margin-bottom:1.5rem;">
        <span class="card-tag" style="background:#E0F2FE; color:#0369A1;">Live Q&A</span>
        <h2 class="serif-title" style="font-size:2.2rem; margin:6px 0;">Ask Your Data</h2>
        <p style="font-size:0.9rem; color:#52525B;">Ask natural language questions. The agent writes and executes code to compute the answer.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.warning("Please upload a CSV file in Overview first.")
        st.stop()

    st.markdown(
        """
    <div class="studio-box">
    """,
        unsafe_allow_html=True,
    )
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. Which category has the highest average value?",
        key="chat_question_input",
    )

    if question:
        try:
            with st.spinner("Writing and executing query code..."):
                chat = chat_agent(question, extract_schema(df), df)
            if chat.get("error"):
                st.warning(f"Could not compute answer: {chat['error']}")
            else:
                ans = chat.get("answer")
                if ans is not None:
                    st.markdown(
                        f"""
                    <div style="background:#F4F4F5; border:1.5px solid #18181B; border-radius:12px; padding:16px; margin:14px 0;">
                        <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#71717A; margin-bottom:4px;">Answer:</div>
                        <div style="font-size:1.15rem; font-weight:700; color:#18181B;">{ans}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                for fig in chat.get("figs", []):
                    st.pyplot(fig)
            with st.expander("🔍 View the Code the Agent Wrote"):
                st.code(chat.get("code", ""), language="python")
        except Exception as e:
            st.error(f"Error answering question: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
