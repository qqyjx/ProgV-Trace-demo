import pytest

from progv.claims import parse_claims
from progv.dsl import parse_program
from progv.interp import Interpreter
from progv.intervene import ALPHA, BETA, DELTA, GAMMA, confidence, p_false
from progv.pipeline import analyze
from progv.typecheck import typecheck


def test_weights_sum_to_one():
    assert ALPHA + BETA + GAMMA + DELTA == pytest.approx(1.0)


def test_replace_plans_the_evidence_consistent_value(cases):
    t = analyze(cases["toy-forged-caption"]).trace
    d = t.deltas["title.location"]
    assert (d.old, d.new, d.changed) == ("Marlow City", "Port Alder", True)


def test_true_elements_get_noop_interventions(cases):
    t = analyze(cases["toy-forged-caption"]).trace
    for eid in ("title.event", "title.time", "ocr.0", "frame.f1", "speech.s1"):
        assert not t.deltas[eid].changed
        assert t.scores[eid].value == 0.0


def test_rewrite_uses_the_time_aligned_quote(cases):
    d = analyze(cases["toy-altered-speech"]).trace.deltas["speech.s2"]
    assert d.new == "We will not open the dam gates tonight." and d.changed


def test_responsibility_matches_the_formula(cases):
    sc = analyze(cases["toy-forged-caption"]).responsibility["title.location"]
    p0, p1 = 1 - 0.95 * 0.1, 0.05
    assert sc.parts["verdict"] == pytest.approx(p0 - p1)
    assert sc.parts["ground"] == 1.0
    assert sc.parts["conf"] == pytest.approx(abs(confidence(p0) - confidence(p1)))
    # repairing the location replaces the false-relation edges and drops the recontextualisation edge
    assert 0.0 < sc.parts["path"] < 1.0
    expect = ALPHA * sc.parts["verdict"] + BETA * sc.parts["path"] + GAMMA * sc.parts["ground"] + DELTA * sc.parts["conf"]
    assert sc.value == pytest.approx(expect)


def test_only_false_elements_are_responsible(cases):
    for cid in ("toy-forged-caption", "toy-altered-speech", "toy-mixed"):
        rep = analyze(cases[cid])
        positive = {k for k, s in rep.responsibility.items() if s.value > 0}
        assert positive == rep.grounded_ids
        assert all(0.0 <= s.value <= 1.0 for s in rep.responsibility.values())


def test_authentic_case_has_zero_responsibility(cases):
    assert all(s.value == 0.0 for s in analyze(cases["toy-authentic-control"]).responsibility.values())


def test_single_repair_in_a_joint_case_scores_lower(cases):
    mixed = analyze(cases["toy-mixed"]).responsibility
    single = analyze(cases["toy-forged-caption"]).responsibility["title.location"].value
    assert 0 < mixed["title.time"].value < single
    assert 0 < mixed["speech.s2"].value < single


def test_rerun_flips_the_verdict_when_the_only_false_element_is_repaired(cases):
    case = cases["toy-altered-speech"]
    els = {e.id: e for e in parse_claims(case)}
    rep = analyze(case)
    pert = Interpreter(case)._pass(typecheck(parse_program(rep.program_text), els), rep.trace.deltas["speech.s2"].overrides())
    assert p_false(rep.trace.base) > 0.5 > p_false(pert)


def test_remove_action_drops_the_element(cases):
    case = cases["toy-forged-caption"]
    els = {e.id: e for e in parse_claims(case)}
    text = analyze(case).program_text.replace("action=replace, target=img)\nr_title_location", "action=remove, target=img)\nr_title_location")
    assert "action=remove" in text
    trace = Interpreter(case).run(typecheck(parse_program(text), els))
    assert trace.deltas["title.location"].new is None
    assert trace.scores["title.location"].value > 0.5 * analyze(case).responsibility["title.location"].value
