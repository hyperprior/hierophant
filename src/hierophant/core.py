"""Hierophant: ask questions about a model, get back only claims that check out."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from hierophant.backends import Backend, Message, backend
from hierophant.claims import Check, Claim, Sentence, Verdict, Verifier
from hierophant.evidence import Explainer, RowEvidence
from hierophant.model import Model

SYSTEM = """You explain the predictions of a machine-learning model to {audience}.

You get the model's prediction for one row and that row's SHAP contributions. Answer \
the question in a few short sentences. Every sentence must make exactly one checkable \
claim about this row, as structured data, using one of these kinds:

  {{"kind": "direction", "feature": F, "effect": "up" | "down"}}
      this row's value of F pushed the output up / down
  {{"kind": "top", "feature": F}}
      F was the single largest contributor
  {{"kind": "rank", "feature": F, "other": G}}
      F contributed more than G (by absolute size)
  {{"kind": "magnitude", "feature": F, "value": V}}
      F contributed about V (signed, in output units)

Use feature names exactly as given. Only claim what the evidence supports; each \
claim will be checked against the model, and sentences whose claims fail are deleted.

Reply with JSON only, no prose around it:
{{"sentences": [{{"text": "...", "claim": {{...}}}}, ...]}}"""

USER = """Model: {model_kind}. Dataset: {dataset}.

{evidence}

Question: {question}"""


@dataclass
class Report:
    question: str
    backend: str
    evidence: RowEvidence
    sentences: list[Sentence]
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    raw: str = field(default="", repr=False)
    error: str | None = None

    @property
    def counts(self) -> Counter:
        return Counter(s.check.verdict for s in self.sentences if s.check)

    @property
    def faithfulness(self) -> float:
        """Share of claims the verifier confirmed. Malformed sentences count against it."""
        n = len(self.sentences)
        return self.counts[Verdict.VERIFIED] / n if n else 0.0

    @property
    def kept(self) -> list[Sentence]:
        return [s for s in self.sentences if s.check and s.check.verdict == Verdict.VERIFIED]

    def __str__(self) -> str:
        if self.error:
            return f"[no explanation: {self.error}]"
        prose = " ".join(s.text for s in self.kept) or "[every claim failed its check]"
        cut = len(self.sentences) - len(self.kept)
        tail = f"\n\n({cut} of {len(self.sentences)} sentences cut by the verifier)" if cut else ""
        return prose + tail

    def details(self) -> str:
        lines = [f"Q: {self.question}", f"backend: {self.backend}"]
        for s in self.sentences:
            mark = {"verified": "✓", "refuted": "✗", "inconclusive": "?", "malformed": "!"}[
                s.check.verdict
            ]
            lines += [f"{mark} {s.text}", f"    {s.check.evidence}"]
        lines.append(
            f"faithfulness {self.faithfulness:.0%} · {self.input_tokens}+{self.output_tokens} "
            f"tokens · {self.seconds:.1f}s"
        )
        return "\n".join(lines)


class Hierophant:
    def __init__(
        self,
        model: object,
        X: pd.DataFrame,
        *,
        llm: str | Backend = "ollama:qwen3:8b",
        target: str = "the target",
        dataset: str = "unnamed",
        class_names: list[str] | None = None,
        categories: dict[str, list[str]] | None = None,
        audience: str = "a data scientist debugging the model",
        top: int | None = 8,
    ):
        self.model = Model(model, X, class_names=class_names)
        self.X = X
        self.target = target
        self.dataset = dataset
        self.llm = backend(llm)
        self.audience = audience
        self.top = top
        self.explainer = Explainer(self.model, target, categories)
        self.verifier = Verifier(self.model)

    @classmethod
    def from_dataset(cls, ds, **kw) -> Hierophant:
        """Build from a ``hierophant.datasets.load(...)`` result."""
        return cls(
            ds.model,
            ds.X,
            target=ds.target,
            dataset=f"{ds.name}: {ds.entry.blurb}",
            class_names=ds.class_names,
            categories=ds.categories,
            **kw,
        )

    def ask(self, question: str, *, row: int) -> Report:
        """One LLM call with pre-computed evidence, then every claim gets checked."""
        t0 = time.perf_counter()
        ev = self.explainer.row(self.X, row)
        messages = [
            Message("system", SYSTEM.format(audience=self.audience)),
            Message(
                "user",
                USER.format(
                    model_kind=type(self.model.raw).__name__,
                    dataset=self.dataset,
                    evidence=ev.as_text(self.top),
                    question=question,
                ),
            ),
        ]
        resp = self.llm.complete(messages, json=self.llm.caps.json_mode)
        report = Report(
            question=question,
            backend=self.llm.name,
            evidence=ev,
            sentences=[],
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            raw=resp.text,
        )
        try:
            report.sentences = parse_sentences(resp.text)
        except ValueError as e:
            report.error = str(e)
        x = self.X.iloc[[row]]
        for s in report.sentences:
            if s.check is None:
                s.check = self.verifier.check(s.claim, x, ev)
        report.seconds = time.perf_counter() - t0
        return report


def parse_sentences(text: str) -> list[Sentence]:
    """Pull the JSON out of an LLM reply, tolerating fences and chatter around it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reply contained no JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"reply JSON did not parse: {e}") from e
    items = data.get("sentences") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("reply JSON has no 'sentences' list")
    out = []
    for item in items:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        claim = item.get("claim")
        if isinstance(claim, dict):
            out.append(Sentence(str(item["text"]), Claim.from_dict(claim)))
        else:
            out.append(
                Sentence(str(item["text"]), None, Check(Verdict.MALFORMED, "no claim attached"))
            )
    return out
