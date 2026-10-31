import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_task(y: pd.Series) -> str:
    if not is_numeric_dtype(y) or y.dtype == bool or y.nunique() <= 10:
        return "classification"
    return "regression"


def run_baseline(df: pd.DataFrame, target: str, time_ordered: bool = False) -> dict:
    data = df.dropna(subset=[target]).copy()
    y = data[target]
    X = data.drop(columns=[target])

    # Drop columns that cannot help: constant columns, and text/ID/date-like columns
    dropped = [
        c for c in X.columns
        if X[c].nunique() <= 1 or (not is_numeric_dtype(X[c]) and X[c].nunique() > 50)
    ]
    X = X.drop(columns=dropped)
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left after cleaning.")
    if len(data) < 30:
        raise ValueError("Need at least 30 rows with a known target.")

    task = detect_task(y)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    split = {"test_size": 0.2, "random_state": 42, "shuffle": not time_ordered}
    if task == "classification" and not time_ordered and y.value_counts().min() >= 2:
        split["stratify"] = y
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, **split)

    if task == "classification":
        models = {
            "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        }
    else:
        models = {
            "Dummy (mean)": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        }

    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("pre", clone(preprocess)), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        fitted[name] = pipe
        if task == "classification":
            rows.append({"Model": name,
                         "Accuracy": round(accuracy_score(y_te, pred), 4),
                         "F1 (macro)": round(f1_score(y_te, pred, average="macro"), 4)})
        else:
            rows.append({"Model": name,
                         "R2": round(r2_score(y_te, pred), 4),
                         "MAE": round(mean_absolute_error(y_te, pred), 4)})

    rf = fitted["Random Forest"]
    names = [n.split("__", 1)[-1] for n in rf.named_steps["pre"].get_feature_names_out()]
    importance = (pd.Series(rf.named_steps["model"].feature_importances_, index=names)
                  .sort_values(ascending=False).head(10))

    return {
        "task": task,
        "results": pd.DataFrame(rows),
        "importance": importance,
        "dropped": dropped,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }