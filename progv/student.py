"""Module F (toy): a distilled student verifier. Two logistic heads, numpy only, trained on teacher outputs.

At prediction time the student sees only the raw case through cheap surface features. It runs no program,
no retrieval and no LLM. It is a toy illustration of program-to-student distillation, not the paper's BERT student.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .claims import Case, Element, parse_claims
from .evidence import aligned_quote, norm
from .pipeline import analyze
from .synth import generate_cases

KINDS = ("title.location", "title.time", "title.event", "ocr", "frame", "speech")
FEATURES = ("has_fact", "agree_frac", "disagree_frac", "has_quote", "token_jaccard", "neg_mismatch", "quote_diff")
GROUND_THRESHOLD = 0.2  # toy cut-off on the predicted responsibility; chosen by hand for this toy data
_NEG = re.compile(r"\b(not|no|nobody|never|n't)\b")


def _kind(el: Element) -> str:
    return el.id if el.id.startswith("title.") else el.source


def element_features(case: Case, el: Element) -> np.ndarray:
    """Surface agreement between an element and the pooled evidence of the case (no program, no retrieval)."""
    f = dict.fromkeys(FEATURES, 0.0)
    if el.source == "speech":
        hit = aligned_quote(el.span, case.evidence)
        if hit:
            q = hit[1]
            a, b = set(norm(el.value).split()), set(norm(q).split())
            f["has_quote"] = 1.0
            f["token_jaccard"] = len(a & b) / len(a | b)
            f["neg_mismatch"] = float(bool(_NEG.search(el.value.lower())) != bool(_NEG.search(q.lower())))
            f["quote_diff"] = float(norm(q) != norm(el.value))
    else:
        vals = [d.facts[el.slot] for d in case.evidence if el.slot in d.facts]
        if vals:
            agree = sum(norm(v) == norm(el.value) for v in vals)
            f["has_fact"] = 1.0
            f["agree_frac"] = agree / len(vals)
            f["disagree_frac"] = 1.0 - agree / len(vals)
    onehot = [float(_kind(el) == k) for k in KINDS]
    return np.array(onehot + [f[k] for k in FEATURES] + [1.0])


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, lr: float = 1.0, iters: int = 3000) -> np.ndarray:
    """Full-batch gradient descent on the (soft-label) cross-entropy; zero init, so fully deterministic."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
    return w


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class StudentPrediction:
    verdict: str
    label_scores: dict[str, float]  # element id -> predicted responsibility
    localized: list[str]


class Student:
    """Head 1 regresses the teacher's responsibility R(u) per element (soft labels).
    Head 2 maps the largest predicted R to the teacher's verdict."""

    def __init__(self) -> None:
        self.w_elem: np.ndarray | None = None
        self.w_case: np.ndarray | None = None

    def fit(self, pairs: Sequence[tuple[Case, "object"]]) -> "Student":
        """`pairs` are (case, teacher Report) tuples; the labels come from the teacher only."""
        X, y = [], []
        for case, rep in pairs:
            for el in parse_claims(case):
                X.append(element_features(case, el))
                y.append(rep.responsibility[el.id].value)
        self.w_elem = fit_logistic(np.array(X), np.array(y))
        Z = np.array([[max(self._scores(c).values()), 1.0] for c, _ in pairs])
        v = np.array([1.0 if r.verdict == "false" else 0.0 for _, r in pairs])
        self.w_case = fit_logistic(Z, v)
        return self

    def _scores(self, case: Case) -> dict[str, float]:
        return {el.id: float(_sigmoid(element_features(case, el) @ self.w_elem)) for el in parse_claims(case)}

    def predict(self, case: Case) -> StudentPrediction:
        if self.w_elem is None or self.w_case is None:
            raise RuntimeError("student is not trained")
        scores = self._scores(case)
        p = float(_sigmoid(np.array([max(scores.values()), 1.0]) @ self.w_case))
        verdict = "false" if p >= 0.5 else "true"
        loc = [e for e, s in scores.items() if s >= GROUND_THRESHOLD] if verdict == "false" else []
        return StudentPrediction(verdict, scores, loc)


def train_student(n_cases: int = 60, seed: int = 0) -> Student:
    """Distil the teacher (mock LLM + DSL + interventions) on `n_cases` generated synthetic cases."""
    cases = generate_cases(n_cases, seed)
    return Student().fit([(c, analyze(c)) for c in cases])


def agreement(student: Student, cases: Sequence[Case]) -> dict[str, float]:
    """Fraction of cases where the student reproduces the teacher's verdict / set of localized elements."""
    v = g = 0
    for c in cases:
        rep, pred = analyze(c), student.predict(c)
        v += pred.verdict == rep.verdict
        g += set(pred.localized) == set(rep.grounded_ids)
    return {"verdict": v / len(cases), "localization": g / len(cases), "n": float(len(cases))}
