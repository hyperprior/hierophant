"""Attribution, computed once and handed over in a form an LLM can read.

The 2023 version pasted raw numpy arrays into the prompt. Here every number is
rounded, labelled and sorted before an LLM sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from hierophant.model import Model


def sig(x: float, digits: int = 3) -> float:
    """Round to significant digits, which is what a reader actually needs."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(f"{x:.{digits}g}")


@dataclass
class Contribution:
    feature: str
    value: object  # the row's value, decoded if categorical
    shap: float


@dataclass
class RowEvidence:
    row: int
    output: int | None
    output_name: str
    prediction: float
    base_value: float
    contributions: list[Contribution]  # sorted by |shap|, largest first

    def shap_of(self, feature: str) -> float:
        for c in self.contributions:
            if c.feature == feature:
                return c.shap
        raise KeyError(feature)

    def as_text(self, top: int | None = None) -> str:
        cs = self.contributions[:top] if top else self.contributions
        lines = [
            f"Row {self.row}. Output explained: {self.output_name}.",
            f"Prediction: {sig(self.prediction)}. Average prediction (SHAP base value): "
            f"{sig(self.base_value)}.",
            "SHAP contributions (they sum to prediction minus base value), largest first:",
        ]
        for c in cs:
            lines.append(f"- {c.feature} = {c.value}: {sig(c.shap):+}")
        if top and len(self.contributions) > top:
            # No total for the rest: given a number, models make claims about it
            # as if it were a feature (10 of 14 refuted claims on first light).
            lines.append(f"({len(self.contributions) - top} smaller features not shown.)")
        return "\n".join(lines)


class Explainer:
    """SHAP for one model, built lazily and cached."""

    def __init__(self, model: Model, target: str, categories: dict[str, list[str]] | None = None):
        self.model = model
        self.target = target
        self.categories = categories or {}
        self._tree: shap.TreeExplainer | None = None

    def _tree_explainer(self) -> shap.TreeExplainer | None:
        if self._tree is None:
            try:
                # Interventional SHAP against a background sample: the same
                # "replace this feature with typical values" notion the verifier uses.
                background = shap.sample(self.model.X, 100, random_state=0)
                out = "probability" if self.model.is_classifier else "raw"
                self._tree = shap.TreeExplainer(
                    self.model.raw,
                    data=background,
                    model_output=out,
                    feature_perturbation="interventional",
                )
            except Exception:
                return None
        return self._tree

    def row(self, X: pd.DataFrame, i: int) -> RowEvidence:
        x = X.iloc[[i]]
        output = self.model.output_for(x)
        tree = self._tree_explainer()
        if tree is not None:
            sv = tree(x)
            values, base = sv.values[0], sv.base_values[0]
            if values.ndim == 2:  # (features, classes)
                values, base = values[:, output], base[output]
        else:
            background = shap.sample(self.model.X, 100, random_state=0)
            sv = shap.Explainer(
                lambda d: self.model.f(pd.DataFrame(d, columns=X.columns), output), background
            )(x.to_numpy())
            values, base = sv.values[0], sv.base_values[0]

        contribs = [
            Contribution(col, self._decode(col, x.iloc[0][col]), float(v))
            for col, v in zip(X.columns, values, strict=True)
        ]
        contribs.sort(key=lambda c: -abs(c.shap))
        return RowEvidence(
            row=i,
            output=output,
            output_name=self.model.output_name(output, self.target),
            prediction=float(self.model.f(x, output)[0]),
            base_value=float(base),
            contributions=contribs,
        )

    def _decode(self, col: str, v: object) -> object:
        labels = self.categories.get(col)
        if labels is not None:
            code = int(v)
            return labels[code] if 0 <= code < len(labels) else "missing"
        return sig(float(v), 4)
