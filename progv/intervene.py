"""Module E (toy): minimal counterfactual interventions and the responsibility score R(u)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .claims import Element
from .evidence import aligned_quote, norm
from .graph import jaccard_distance

if TYPE_CHECKING:
    from .interp import State

# Weights of Eq. (resp) in the paper: verdict, path, grounding, confidence.
ALPHA, BETA, GAMMA, DELTA = 0.4, 0.3, 0.2, 0.1
STRENGTH = {"contradicted": 0.9, "unknown": 0.15, "supported": 0.0}
PRIOR = 0.05
VERDICT_THRESHOLD = 0.5


@dataclass(frozen=True)
class Delta:
    """A planned perturbation of one element. `new=None` means the element is removed."""
    element: str
    action: str
    old: str
    new: str | None
    changed: bool
    note: str = ""

    def overrides(self) -> dict[str, str | None]:
        return {self.element: self.new} if self.changed else {}


@dataclass(frozen=True)
class Score:
    element: str
    value: float
    parts: dict[str, float]  # verdict, path, ground, conf


def p_false(state: "State") -> float:
    """Noisy-OR aggregation of per-element sub-claim statuses into P(narrative is false)."""
    keep = 1.0 - PRIOR
    for status in state.element_status().values():
        keep *= 1.0 - STRENGTH[status]
    return 1.0 - keep


def confidence(p: float) -> float:
    return abs(2.0 * p - 1.0)


def grounding(state: "State") -> frozenset[str]:
    return frozenset(e for e, s in state.element_status().items() if s == "contradicted")


def plan_delta(el: Element, action: str, target: tuple[str, str], base: "State") -> Delta:
    """Plan the minimal fact-flipping edit: the evidence-consistent counterpart of `el`."""
    kind, payload = target
    new: str | None
    note = ""
    if action == "remove":
        new, note = None, "element removed"
    elif kind == "str":
        new = payload
    else:
        docs = base.env[payload]
        if action == "rewrite":
            hit = aligned_quote(el.span, docs) if el.span else None
            new = hit[1] if hit else el.value
        else:
            vals = [d.facts[el.slot] for d in docs if el.slot in d.facts]
            new = Counter(vals).most_common(1)[0][0] if vals else el.value
        if new == el.value:
            note = "no evidence-consistent counterpart differs from the current value"
    changed = new is None or norm(new) != norm(el.value)
    return Delta(el.id, action, el.value, new, changed, note)


def responsibility(base: "State", pert: "State", element: str) -> Score:
    """R(u) = a*d_verdict + b*d_path + g*d_ground + d*d_conf (Jaccard distance on edge and grounding sets)."""
    p0, p1 = p_false(base), p_false(pert)
    parts = {
        "verdict": abs(p0 - p1),
        "path": jaccard_distance(base.graph.edges, pert.graph.edges),
        "ground": jaccard_distance(set(grounding(base)), set(grounding(pert))),
        "conf": abs(confidence(p0) - confidence(p1)),
    }
    value = ALPHA * parts["verdict"] + BETA * parts["path"] + GAMMA * parts["ground"] + DELTA * parts["conf"]
    return Score(element, value, parts)
