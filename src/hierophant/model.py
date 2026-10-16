"""One view of whatever model the user brings: a function from rows to a number.

Explanations talk about a single scalar output. For a regressor that's the
prediction. For a classifier it's the probability of one class (the positive
class for binary, the predicted class otherwise, or one the user picks).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


class Model:
    def __init__(
        self,
        model: object,
        X: pd.DataFrame,
        *,
        output_class: int | None = None,
        class_names: list[str] | None = None,
        predict: Callable[[pd.DataFrame], np.ndarray] | None = None,
    ):
        self.raw = model
        self.X = X
        self.class_names = class_names
        self.is_classifier = hasattr(model, "predict_proba") and predict is None
        if self.is_classifier and output_class is None:
            n_classes = len(getattr(model, "classes_", [0, 1]))
            output_class = 1 if n_classes == 2 else None
        self.output_class = output_class
        self._predict = predict

    def output_for(self, row: pd.DataFrame) -> int | None:
        """Which class's probability we're explaining for this row."""
        if not self.is_classifier:
            return None
        if self.output_class is not None:
            return self.output_class
        return int(np.argmax(self.raw.predict_proba(row)[0]))

    def f(self, X: pd.DataFrame, output: int | None = None) -> np.ndarray:
        """The scalar we explain, for every row of X."""
        if self._predict is not None:
            return np.asarray(self._predict(X), dtype=float)
        if self.is_classifier:
            k = self.output_class if output is None else output
            return self.raw.predict_proba(X)[:, k]
        return np.asarray(self.raw.predict(X), dtype=float)

    def output_name(self, output: int | None, target: str) -> str:
        if output is None:
            return f"predicted {target}"
        label = self.class_names[output] if self.class_names else str(output)
        return f"probability that {target} is {label!r}"
