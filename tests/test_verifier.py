import json

import pytest

import hierophant as hp
from hierophant.claims import Claim, Verdict


def reply(*sentences):
    return json.dumps({"sentences": [{"text": t, "claim": c} for t, c in sentences]})


@pytest.fixture
def h(diabetes):
    return hp.Hierophant.from_dataset(diabetes, llm=hp.Replay([]))


def truth(h, row=0):
    """What a perfect explainer would say about this row."""
    ev = h.explainer.row(h.X, row)
    top, second = ev.contributions[0], ev.contributions[1]
    return ev, top, second


def test_shap_is_additive(h):
    ev = h.explainer.row(h.X, 0)
    assert ev.base_value + sum(c.shap for c in ev.contributions) == pytest.approx(ev.prediction)


def test_true_claims_verify(h):
    ev, top, second = truth(h)
    up = "up" if top.shap > 0 else "down"
    h.llm = hp.Replay(
        [
            reply(
                ("top pushes", {"kind": "direction", "feature": top.feature, "effect": up}),
                ("top is top", {"kind": "top", "feature": top.feature}),
                ("rank", {"kind": "rank", "feature": top.feature, "other": second.feature}),
                ("size", {"kind": "magnitude", "feature": top.feature, "value": top.shap * 1.1}),
            )
        ]
    )
    r = h.ask("why?", row=0)
    assert [s.check.verdict for s in r.sentences] == [Verdict.VERIFIED] * 4
    assert r.faithfulness == 1.0


def test_false_claims_refute(h):
    ev, top, second = truth(h)
    wrong = "down" if top.shap > 0 else "up"
    h.llm = hp.Replay(
        [
            reply(
                ("wrong way", {"kind": "direction", "feature": top.feature, "effect": wrong}),
                ("wrong top", {"kind": "top", "feature": second.feature}),
                ("wrong rank", {"kind": "rank", "feature": second.feature, "other": top.feature}),
                ("wrong size", {"kind": "magnitude", "feature": top.feature, "value": -top.shap}),
                ("made up", {"kind": "top", "feature": "shoe_size"}),
            )
        ]
    )
    r = h.ask("why?", row=0)
    assert all(s.check.verdict == Verdict.REFUTED for s in r.sentences)
    assert r.faithfulness == 0.0
    assert "no feature named 'shoe_size'" in r.sentences[-1].check.evidence


def test_report_prints_only_verified(h):
    ev, top, second = truth(h)
    h.llm = hp.Replay(
        [
            reply(
                ("Kept.", {"kind": "top", "feature": top.feature}),
                ("Cut.", {"kind": "top", "feature": second.feature}),
            )
        ]
    )
    out = str(h.ask("why?", row=0))
    assert out.startswith("Kept.") and "Cut." not in out
    assert "1 of 2 sentences cut" in out


def test_tolerates_fences_thinking_and_bad_items(h):
    ev, top, _ = truth(h)
    text = (
        "<think>hmm {not json}</think>```json\n"
        + reply(
            ("ok", {"kind": "top", "feature": top.feature}),
        ).replace("]}", ', {"text": "no claim"}, {"claim": {}}, "junk"]}')
        + "\n```"
    )
    h.llm = hp.Replay([text])
    r = h.ask("why?", row=0)
    assert [s.check.verdict for s in r.sentences] == [Verdict.VERIFIED, Verdict.MALFORMED]


def test_unparseable_reply_is_an_error_not_a_crash(h):
    h.llm = hp.Replay(["I think bmi matters a lot."])
    r = h.ask("why?", row=0)
    assert r.error and r.sentences == [] and r.faithfulness == 0.0


def test_classifier_direction_uses_explained_class(cancer):
    h = hp.Hierophant.from_dataset(cancer, llm=hp.Replay([]))
    ev = h.explainer.row(h.X, 3)
    assert ev.output == 1 and "benign" in ev.output_name
    top = ev.contributions[0]
    claim = Claim("direction", top.feature, effect="up" if top.shap > 0 else "down")
    assert h.verifier.check(claim, h.X.iloc[[3]], ev).verdict == Verdict.VERIFIED


def test_prompt_carries_rounded_evidence(h):
    h.llm = hp.Replay([reply()])
    h.ask("why is this low?", row=0)
    system, user = h.llm.seen[0]
    assert "JSON" in system.content and "why is this low?" in user.content
    assert "e-0" not in user.content  # no raw float dumps
