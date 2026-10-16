"""A shelf of public tabular datasets, each paired with a fitted model.

Two tiers:

- ``bundled``: ship with scikit-learn, no network. Tests use only these.
- ``download``: fetched once from OpenML / StatLib and cached under
  ``~/.cache/hierophant``. Examples and the benchmark use these.

Every entry records where the data comes from and its license, so nothing gets
published that we don't have the right to publish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as skd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

CACHE = Path.home() / ".cache" / "hierophant"


@dataclass(frozen=True)
class Entry:
    name: str
    task: str  # "regression" | "binary" | "multiclass"
    tier: str  # "bundled" | "download"
    source: str
    license: str
    blurb: str


REGISTRY: dict[str, Entry] = {
    e.name: e
    for e in [
        Entry(
            "diabetes",
            "regression",
            "bundled",
            "Efron et al. (2004), via scikit-learn",
            "BSD-3-Clause (scikit-learn)",
            "Disease progression one year after baseline, from 10 clinical measurements.",
        ),
        Entry(
            "breast_cancer",
            "binary",
            "bundled",
            "UCI Breast Cancer Wisconsin (Diagnostic), via scikit-learn",
            "CC BY 4.0",
            "Malignant vs benign tumours from 30 correlated cell-nucleus measurements.",
        ),
        Entry(
            "wine",
            "multiclass",
            "bundled",
            "UCI Wine, via scikit-learn",
            "CC BY 4.0",
            "Cultivar of origin for 178 Italian wines from 13 chemical measurements.",
        ),
        Entry(
            "california_housing",
            "regression",
            "download",
            "Pace & Barry (1997), StatLib, via scikit-learn",
            "Public domain (StatLib)",
            "Median house value per California block group, 1990 census.",
        ),
        Entry(
            "adult",
            "binary",
            "download",
            "UCI Adult / Census Income, OpenML id 1590",
            "CC BY 4.0",
            "Whether income exceeds $50k/yr, from the 1994 US census.",
        ),
        Entry(
            "bike_sharing",
            "regression",
            "download",
            "UCI Bike Sharing (hourly), OpenML id 42713",
            "CC BY 4.0",
            "Hourly rental count in Washington DC from time and weather.",
        ),
        Entry(
            "credit_g",
            "binary",
            "download",
            "UCI Statlog German Credit, OpenML id 31",
            "CC BY 4.0",
            "Good vs bad credit risk for 1000 loan applicants.",
        ),
    ]
}

_OPENML_IDS = {"adult": 1590, "bike_sharing": 42713, "credit_g": 31}


@dataclass
class Dataset:
    """Data, a fitted model, and enough metadata for an LLM to talk about it."""

    entry: Entry
    X: pd.DataFrame
    y: pd.Series
    model: object
    X_test: pd.DataFrame
    y_test: pd.Series
    target: str
    class_names: list[str] | None = None
    # Integer-coded categorical columns: column -> list of labels (code = index).
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def task(self) -> str:
        return self.entry.task

    def score(self) -> float:
        """R² for regression, accuracy for classification, on the held-out split."""
        return float(self.model.score(self.X_test, self.y_test))


def names(tier: str | None = None) -> list[str]:
    return [n for n, e in REGISTRY.items() if tier is None or e.tier == tier]


def load(name: str, *, seed: int = 0, n_estimators: int = 200) -> Dataset:
    """Load a dataset from the shelf and fit a gradient-boosted tree model on it."""
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; have {', '.join(REGISTRY)}")
    entry = REGISTRY[name]
    X, y, target, class_names = _fetch(name)
    X, categories = _encode_categoricals(X)
    if class_names is not None:
        y = pd.Series(pd.Categorical(y, categories=class_names).codes, index=y.index, name=y.name)

    stratify = y if entry.task != "regression" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=stratify
    )
    params = dict(n_estimators=n_estimators, max_depth=3, learning_rate=0.05, random_state=seed)
    model = (
        GradientBoostingRegressor(**params)
        if entry.task == "regression"
        else GradientBoostingClassifier(**params)
    )
    model.fit(X_tr, y_tr)
    return Dataset(
        entry=entry,
        X=X_tr.reset_index(drop=True),
        y=y_tr.reset_index(drop=True),
        model=model,
        X_test=X_te.reset_index(drop=True),
        y_test=y_te.reset_index(drop=True),
        target=target,
        class_names=class_names,
        categories=categories,
    )


def _fetch(name: str) -> tuple[pd.DataFrame, pd.Series, str, list[str] | None]:
    if name == "diabetes":
        b = skd.load_diabetes(as_frame=True, scaled=False)
        return b.data, b.target.rename("progression"), "disease progression", None
    if name == "breast_cancer":
        b = skd.load_breast_cancer(as_frame=True)
        names = [str(n) for n in b.target_names]
        return b.data, b.target.map(dict(enumerate(names))), "diagnosis", names
    if name == "wine":
        b = skd.load_wine(as_frame=True)
        names = [str(n) for n in b.target_names]
        return b.data, b.target.map(dict(enumerate(names))), "cultivar", names
    if name == "california_housing":
        b = skd.fetch_california_housing(as_frame=True, data_home=CACHE)
        return b.data, b.target, "median house value ($100k)", None

    b = skd.fetch_openml(data_id=_OPENML_IDS[name], as_frame=True, data_home=CACHE)
    X, y = b.data, b.target
    if name == "bike_sharing":
        return X, y.astype(float), "rentals per hour", None
    if name == "adult":
        X = X.drop(columns=["fnlwgt"], errors="ignore")
        return X, y.astype(str), "income", ["<=50K", ">50K"]
    if name == "credit_g":
        return X, y.astype(str), "credit risk", ["bad", "good"]
    raise AssertionError(name)


def _encode_categoricals(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Tree models want numbers. Integer-code categoricals and keep the labels."""
    X = X.copy()
    categories = {}
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == object:
            cat = X[col].astype("category")
            categories[col] = [str(c) for c in cat.cat.categories]
            X[col] = cat.cat.codes.astype(np.int64)  # missing -> -1
        else:
            X[col] = X[col].astype(float)
    return X, categories
