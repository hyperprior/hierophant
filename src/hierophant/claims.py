"""Claims an explanation makes about one row, and the checks that keep it honest.

Each sentence an LLM writes carries one of these. The verifier tests it against
the model, and does not take the LLM's word for anything.

Claim types (all about a single row):

- ``direction``: this row's value of ``feature`` pushed the output up or down.
  Checked by intervention: swap the feature for typical values from the data
  and see which way the output moves.
- ``top``: ``feature`` was the single largest contributor (by |SHAP|).
- ``rank``: ``feature`` contributed more than ``other`` (by |SHAP|).
- ``magnitude``: ``feature`` contributed about ``value`` (signed, output units).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from hierophant.evidence import RowEvidence, sig
from hierophant.model import Model

KINDS = ("direction", "top", "rank", "magnitude")


class Verdict(StrEnum):
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    MALFORMED = "malformed"


@dataclass
class Claim:
    kind: str
    feature: str
    effect: str | None = None  # direction: "up" | "down"
    other: str | None = None  # rank
    value: float | None = None  # magnitude

    @classmethod
    def from_dict(cls, d: dict) -> Claim:
        return cls(
            kind=str(d.get("kind", "")),
            feature=str(d.get("feature", "")),
            effect=d.get("effect"),
            other=d.get("other"),
            value=_float_or_none(d.get("value")),
        )


@dataclass
class Check:
    verdict: Verdict
    evidence: str


@dataclass
class Sentence:
    text: str
    claim: Claim | None
    check: Check | None = None
    meta: dict = field(default_factory=dict)


class Verifier:
    def __init__(self, model: Model, *, background: int = 100, seed: int = 0):
        self.model = model
        rng = np.random.default_rng(seed)
        n = min(background, len(model.X))
        self.background = model.X.iloc[rng.choice(len(model.X), n, replace=False)]
        self._scale: dict[int | None, float] = {}

    def scale(self, output: int | None) -> float:
        """Typical spread of the output; effects smaller than 1% of it are noise."""
        if output not in self._scale:
            self._scale[output] = float(np.std(self.model.f(self.background, output))) or 1.0
        return self._scale[output]

    def intervene(self, x: pd.DataFrame, feature: str, output: int | None) -> float:
        """f(x) minus the average f(x) with `feature` swapped for background values."""
        swapped = pd.concat([x] * len(self.background), ignore_index=True)
        swapped[feature] = self.background[feature].to_numpy()
        base = float(self.model.f(x, output)[0])
        return base - float(np.mean(self.model.f(swapped, output)))

    def check(self, claim: Claim | None, x: pd.DataFrame, ev: RowEvidence) -> Check:
        if claim is None:
            return Check(Verdict.MALFORMED, "sentence carried no claim")
        if claim.kind not in KINDS:
            return Check(Verdict.MALFORMED, f"unknown claim kind {claim.kind!r}")
        known = set(x.columns)
        for f in (claim.feature, claim.other) if claim.kind == "rank" else (claim.feature,):
            if f not in known:
                return Check(Verdict.REFUTED, f"no feature named {f!r} in this model")

        s = ev.shap_of(claim.feature)
        if claim.kind == "direction":
            if claim.effect not in ("up", "down"):
                return Check(
                    Verdict.MALFORMED, f"direction effect must be up/down, got {claim.effect!r}"
                )
            delta = self.intervene(x, claim.feature, ev.output)
            tol = 0.01 * self.scale(ev.output)
            note = (
                f"swapping {claim.feature} for typical values moves the output by "
                f"{sig(-delta):+} (SHAP {sig(s):+})"
            )
            if abs(delta) <= tol:
                return Check(Verdict.INCONCLUSIVE, note + "; too small to call")
            pushed_up = delta > 0
            ok = pushed_up == (claim.effect == "up")
            return Check(Verdict.VERIFIED if ok else Verdict.REFUTED, note)

        if claim.kind == "top":
            top = ev.contributions[0]
            ok = top.feature == claim.feature
            return Check(
                Verdict.VERIFIED if ok else Verdict.REFUTED,
                f"largest contributor is {top.feature} ({sig(top.shap):+}); "
                f"{claim.feature} is {sig(s):+}",
            )

        if claim.kind == "rank":
            o = ev.shap_of(claim.other)
            ok = abs(s) > abs(o)
            return Check(
                Verdict.VERIFIED if ok else Verdict.REFUTED,
                f"|SHAP| {claim.feature} = {sig(abs(s))}, {claim.other} = {sig(abs(o))}",
            )

        # magnitude
        if claim.value is None:
            return Check(Verdict.MALFORMED, "magnitude claim without a value")
        tol = max(0.25 * abs(s), 0.05 * self.scale(ev.output))
        ok = abs(claim.value - s) <= tol and np.sign(claim.value) == np.sign(s)
        return Check(
            Verdict.VERIFIED if ok else Verdict.REFUTED,
            f"claimed {sig(claim.value):+}, SHAP is {sig(s):+} (tolerance ±{sig(tol)})",
        )


def _float_or_none(v: object) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
