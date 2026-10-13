import numpy as np
import pytest

import progv.interp
from progv.pipeline import analyze
from progv.student import Student, agreement, element_features, fit_logistic, train_student
from progv.synth import generate_cases


@pytest.fixture(scope="module")
def student():
    return train_student(60, seed=0)


def test_logistic_fit_learns_a_separable_rule():
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], float)
    w = fit_logistic(X, np.array([0, 0, 0, 1.0]), iters=4000)
    p = 1 / (1 + np.exp(-X @ w))
    assert p[3] > 0.5 > max(p[:3])


def test_fit_is_deterministic():
    assert np.array_equal(train_student(20, seed=2).w_elem, train_student(20, seed=2).w_elem)


def test_untrained_student_refuses(cases):
    with pytest.raises(RuntimeError):
        Student().predict(cases["toy-mixed"])


def test_student_reproduces_teacher_on_toy_cases(student, cases):
    for c in cases.values():
        rep, pred = analyze(c), student.predict(c)
        assert pred.verdict == rep.verdict, c.id
        assert set(pred.localized) == rep.grounded_ids, c.id


def test_student_generalises_to_heldout_generated_cases(student):
    res = agreement(student, generate_cases(60, seed=11))
    assert res["verdict"] >= 0.9 and res["localization"] >= 0.9


def test_student_needs_no_program_execution(student, cases, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("student must not run the interpreter")

    monkeypatch.setattr(progv.interp.Interpreter, "run", boom)
    monkeypatch.setattr(progv.interp.Interpreter, "_pass", boom)
    assert student.predict(cases["toy-forged-caption"]).verdict == "false"


def test_features_flag_disagreement(cases):
    from progv.claims import parse_claims
    els = {e.id: e for e in parse_claims(cases["toy-forged-caption"])}
    names = ["has_fact", "agree_frac", "disagree_frac", "has_quote", "token_jaccard", "neg_mismatch", "quote_diff"]
    f = dict(zip(names, element_features(cases["toy-forged-caption"], els["title.location"])[6:13]))
    assert f["disagree_frac"] == 1.0 and f["agree_frac"] == 0.0
    f = dict(zip(names, element_features(cases["toy-forged-caption"], els["title.time"])[6:13]))
    assert f["agree_frac"] == 1.0
