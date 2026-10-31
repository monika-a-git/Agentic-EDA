# 📊 Agentic AI EDA & ML Baseline Pipeline

An end-to-end Agentic Exploratory Data Analysis (EDA) and Machine Learning pipeline powered by multi-provider LLM orchestration with automatic fallback between **Groq**, **Google Gemini**, and **OpenAI**.

---

## 🌟 Key Features

1. **Data Health & Profiling**:
   - Computes an automated 0–100 Data Health Score.
   - Detects missing value rates, duplicate rows, constant columns, and potential IQR outliers.

2. **Multi-Agent EDA System**:
   - **Coder Agent**: Automatically generates safe, sandboxed Python plotting code (histograms, correlation heatmaps, missing value bar plots, time trends) with self-healing retry logic.
   - **Analyst Agent**: Synthesizes metadata, descriptive statistics, correlations, and data quality issues into a structured business executive summary report.
   - **Verifier Agent ("Fact Checker")**: Extracts exact numeric claims from the Analyst's report and verifies them directly against the actual dataset with executable Pandas assertions (showing ✅ Verified, ❌ Mismatch, or ⚠️ Could not check).

3. **🤖 Automated Machine Learning Baseline**:
   - Auto-detects whether the prediction task is **Classification** or **Regression**.
   - Cleans features, imputes missing values, scales numbers, and one-hot encodes categories.
   - Compares baseline models: Dummy vs. Linear/Logistic Regression vs. Random Forest.
   - Computes top feature importances.
   - Uses an LLM to explain the model results, evaluate over-fitting / data leakage risks, and suggest next steps.

4. **💬 Interactive Natural Language Data Chat**:
   - Ask any question about your dataset (e.g., *"Which category had the highest sales?"*).
   - Generates and executes safe data manipulation code to return direct answers and optional plots.

5. **🛡️ Resilient Multi-Provider Fallback**:
   - Tries fast, free inference from **Groq** (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`).
   - Automatically fails over to **Google Gemini** (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`).
   - Supports **OpenAI** (`gpt-4o-mini`) as an additional backup.
   - If one provider hits rate limits or downtime, the next provider seamlessly handles the request.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.streamlit/secrets.toml` in the project root:

```toml
GROQ_API_KEY = "gsk_your_groq_key_here"
GEMINI_API_KEY = "your_gemini_key_here"
```

> **Where to get keys:**
> - **Groq** (Free): [console.groq.com](https://console.groq.com)
> - **Gemini** (Free): [aistudio.google.com](https://aistudio.google.com)

*Note: You can also enter API keys directly in the app's sidebar during runtime.*

### 3. Run the App

```bash
streamlit run app.py
```

