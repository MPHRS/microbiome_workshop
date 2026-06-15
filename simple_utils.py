from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

META_COLS = {
    "run", "bioproject", "source", "layout", "year", "country",
    "sequencing_platform", "instrument", "instrument_model",
    "library_strategy", "library_source", "health_status",
    "healthy", "K02_caries", "K05_gingivitis_periodontitis",
    "K02", "K05", "is_healthy", "is_k05", "is_k02",
    "disease_label", "group",
}
UNWANTED_PATHWAY_COLS = {"UNMAPPED", "UNINTEGRATED"}
GROUP_ORDER = ["Healthy", "Caries (K02)", "Periodontitis (K05)", "K02+K05"]


class DataBundle(NamedTuple):
    meta: pd.DataFrame
    tax: pd.DataFrame
    path: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV table and index it by run."""
    df = pd.read_csv(path, low_memory=False)
    if "run" not in df.columns:
        raise ValueError(f"Column 'run' is required in {path}")
    df["run"] = df["run"].astype(str).str.strip()
    return df.drop_duplicates("run").set_index("run").sort_index()


def feature_columns(df: pd.DataFrame, feature_type: str) -> list[str]:
    """Keep microbiome feature columns and remove metadata columns."""
    cols = [c for c in df.columns if c not in META_COLS]
    if feature_type.upper() == "PATH":
        cols = [c for c in cols if c not in UNWANTED_PATHWAY_COLS and "|" not in c]
    return cols


def load_data(tax_path, path_path, metadata_path, pathway_scale: float = 1e6) -> DataBundle:
    """Load metadata, TAX and PATH tables and align them by run."""
    meta = read_table(metadata_path)
    tax_raw = read_table(tax_path)
    path_raw = read_table(path_path)

    required = {"bioproject", "healthy", "K02_caries", "K05_gingivitis_periodontitis"}
    missing = sorted(required - set(meta.columns))
    if missing:
        raise ValueError(f"Metadata columns are missing: {missing}")

    runs = meta.index.intersection(tax_raw.index).intersection(path_raw.index)
    if runs.empty:
        raise ValueError("No common runs between metadata, TAX and PATH tables")

    meta = meta.loc[runs].copy()
    tax_raw = tax_raw.loc[runs]
    path_raw = path_raw.loc[runs]

    for col in ["healthy", "K02_caries", "K05_gingivitis_periodontitis"]:
        meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0).astype(int)

    meta["bioproject"] = meta["bioproject"].astype(str)
    meta["is_healthy"] = (meta["healthy"] == 1).astype(int)
    meta["is_k02"] = (meta["K02_caries"] == 1).astype(int)
    meta["is_k05"] = (meta["K05_gingivitis_periodontitis"] == 1).astype(int)

    meta["group"] = np.select(
        [
            (meta["is_k02"] == 1) & (meta["is_k05"] == 1),
            meta["is_k02"] == 1,
            meta["is_k05"] == 1,
            meta["is_healthy"] == 1,
        ],
        ["K02+K05", "Caries (K02)", "Periodontitis (K05)", "Healthy"],
        default="Unknown",
    )

    tax = tax_raw[feature_columns(tax_raw, "TAX")].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    path = path_raw[feature_columns(path_raw, "PATH")].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    path = path / pathway_scale

    return DataBundle(meta=meta, tax=tax, path=path)


def make_target(meta: pd.DataFrame, target: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Keep target disease and healthy controls only."""
    disease_col = {"K05": "is_k05", "K02": "is_k02"}[target.upper()]
    meta_task = meta.loc[(meta[disease_col] == 1) | (meta["is_healthy"] == 1)].copy()
    y = meta_task[disease_col].to_numpy(dtype=int)
    return y, meta_task


def feature_matrix(bundle: DataBundle, feature_set: str, runs) -> pd.DataFrame:
    """Return TAX, PATH or TAX+PATH for selected runs."""
    runs = list(runs)
    feature_set = feature_set.upper()
    if feature_set == "TAX":
        return bundle.tax.loc[runs]
    if feature_set == "PATH":
        return bundle.path.loc[runs]
    if feature_set == "TAX+PATH":
        return pd.concat(
            [bundle.tax.loc[runs].add_prefix("TAX__"), bundle.path.loc[runs].add_prefix("PATH__")],
            axis=1,
        )
    raise ValueError("feature_set must be 'TAX', 'PATH' or 'TAX+PATH'")


def clr_transform(X, pseudocount: float = 1e-6) -> np.ndarray:
    """CLR transformation through scikit-bio."""
    from skbio.stats.composition import closure, clr

    X = np.asarray(X, dtype=float)
    return np.asarray(clr(closure(X + pseudocount)), dtype=float)


def alpha_diversity_table(X: pd.DataFrame) -> pd.DataFrame:
    """Alpha diversity through skbio.diversity.alpha."""
    from skbio.diversity.alpha import observed_features, shannon, simpson

    values = X.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "observed_features": [observed_features(row) for row in values],
            "shannon": [shannon(row) for row in values],
            "simpson": [simpson(row) for row in values],
        },
        index=X.index,
    )


def pca_scores_from_clr(X: pd.DataFrame, n_components: int = 2):
    """CLR-transform the matrix and return PCA scores."""
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(clr_transform(X))
    scores = pd.DataFrame(scores, index=X.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    return scores, pca.explained_variance_ratio_


def prepare_train_test(bundle: DataBundle, train_runs, test_runs, feature_set: str, max_features: int = 2000):
    """Select train features, apply CLR and scale train/test matrices."""
    x_train_raw = feature_matrix(bundle, feature_set, train_runs)
    x_test_raw = feature_matrix(bundle, feature_set, test_runs)

    selector = VarianceThreshold(0.0).fit(x_train_raw)
    cols = pd.Index(x_train_raw.columns[selector.get_support()])
    if len(cols) == 0:
        raise ValueError("No non-constant features in this training split")

    if len(cols) > max_features:
        prevalence = (x_train_raw[cols] > 0).mean(axis=0)
        cols = prevalence.sort_values(ascending=False).index[:max_features]

    x_train = clr_transform(x_train_raw[cols])
    x_test = clr_transform(x_test_raw[cols])
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test), list(cols)


def fit_logistic_regression(x_train, y_train, random_state: int = 42):
    """Fit ridge logistic regression with a small inner CV grid when possible."""
    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )

    n_inner = min(3, int(y_train.sum()), int((y_train == 0).sum()))
    if n_inner < 2:
        return model.fit(x_train, y_train), {"C": None}

    grid = GridSearchCV(
        model,
        {"C": [0.01, 0.1, 1, 10, 100]},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state),
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    return grid.best_estimator_, grid.best_params_


def classification_metrics(y_true, probability) -> dict[str, float]:
    """AUC, MCC and balanced accuracy."""
    if len(np.unique(y_true)) < 2:
        return {"AUC": np.nan, "MCC": np.nan, "BACC": np.nan}
    pred = (probability >= 0.5).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, probability)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "BACC": float(balanced_accuracy_score(y_true, pred)),
    }


def summarize_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD for successful CV folds."""
    ok = rows[rows["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()

    out = {"n_splits": len(ok)}
    for metric in ["AUC", "MCC", "BACC"]:
        out[f"mean_{metric}"] = ok[metric].mean()
        out[f"sd_{metric}"] = ok[metric].std(ddof=1)
    return pd.DataFrame([out])
