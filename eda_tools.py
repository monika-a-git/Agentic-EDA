import pandas as pd
import numpy as np


def extract_schema(df: pd.DataFrame) -> str:
    """Extract dataset metadata including shapes, data types, null counts, and sample values."""
    lines = [
        f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns",
        "Columns and Data Types:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        n_unique = int(df[col].nunique())
        non_null = df[col].dropna()
        if len(non_null) > 0:
            samples = non_null.head(3).tolist()
            sample_str = ", ".join(repr(s) for s in samples)
        else:
            sample_str = "All NaN"
        lines.append(
            f"- `{col}` ({dtype}): {null_count} nulls ({null_pct:.1f}%), {n_unique} unique. Examples: [{sample_str}]"
        )
    return "\n".join(lines)


def build_summary(df: pd.DataFrame) -> str:
    """Produce summary statistics with one row per column."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {"column": col, "dtype": str(s.dtype), "non_null": int(s.count())}
        if pd.api.types.is_numeric_dtype(s):
            row["mean"] = round(float(s.mean()), 4) if pd.notnull(s.mean()) else None
            row["std"] = round(float(s.std()), 4) if pd.notnull(s.std()) else None
            row["min"] = round(float(s.min()), 4) if pd.notnull(s.min()) else None
            row["median"] = round(float(s.median()), 4) if pd.notnull(s.median()) else None
            row["max"] = round(float(s.max()), 4) if pd.notnull(s.max()) else None
        else:
            vc = s.value_counts()
            row["top_value"] = str(vc.index[0]) if not vc.empty else None
            row["top_freq"] = int(vc.iloc[0]) if not vc.empty else None
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


def top_correlations(df: pd.DataFrame, n: int = 8) -> str:
    """Compute and format the strongest pairwise numeric correlations."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns for correlation analysis."
    corr_matrix = numeric_df.corr()
    pairs = []
    cols = numeric_df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1, col2 = cols[i], cols[j]
            val = corr_matrix.loc[col1, col2]
            if pd.notnull(val):
                pairs.append((col1, col2, val, abs(val)))
    if not pairs:
        return "No valid correlations calculated."
    pairs.sort(key=lambda x: x[3], reverse=True)
    top_pairs = pairs[:n]
    lines = [f"- {c1} & {c2}: Pearson r = {r:.4f}" for c1, c2, r, _ in top_pairs]
    return "\n".join(lines)


def health_report(df: pd.DataFrame) -> dict:
    rows, cols = df.shape
    if rows == 0 or cols == 0:
        return {"score": 0, "issues": ["Dataset is empty"]}
    missing_pct = df.isnull().sum().sum() / (rows * cols) * 100
    dup_pct = df.duplicated().sum() / rows * 100
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # Outliers: share of values outside 1.5 * IQR, averaged over numeric columns
    outlier_rates = []
    for c in df.select_dtypes(include="number").columns:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        bad = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).mean()
        outlier_rates.append(bad * 100)
    outlier_pct = sum(outlier_rates) / len(outlier_rates) if outlier_rates else 0

    score = 100
    score -= min(40, missing_pct * 2)
    score -= min(20, dup_pct * 2)
    score -= min(15, len(constant_cols) / cols * 100)
    score -= min(25, outlier_pct * 2)

    issues = []
    if missing_pct > 0:
        issues.append(f"{missing_pct:.1f}% of all cells are missing")
    if dup_pct > 0:
        issues.append(f"{dup_pct:.1f}% of rows are duplicates")
    if constant_cols:
        issues.append(f"Constant columns (no information): {constant_cols}")
    if outlier_pct > 1:
        issues.append(f"About {outlier_pct:.1f}% of numeric values look like outliers")

    return {"score": round(max(score, 0)), "issues": issues}
