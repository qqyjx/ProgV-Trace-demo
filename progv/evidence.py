"""Evidence retrieval and the element-vs-evidence judge (toy stand-ins for Module C/D classifiers)."""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .claims import Element, EvidenceDoc


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def _tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if len(t) > 3}


def retrieve(docs: Sequence[EvidenceDoc], etype: str, query: str) -> list[EvidenceDoc]:
    """Docs of the requested type whose query handle equals `query`, or that share >= 2 content words."""
    q = _tokens(query)
    return [d for d in docs if d.type == etype
            and (query in d.queries or len(q & _tokens(d.text)) >= 2)]


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def aligned_quote(span: tuple[float, float], docs: Sequence[EvidenceDoc],
                  min_iou: float = 0.3) -> tuple[EvidenceDoc, str] | None:
    """Temporal alignment: the evidence quote whose time span overlaps `span` the most."""
    best, best_iou = None, min_iou
    for d in docs:
        for t0, t1, text in d.quotes:
            v = iou(span, (t0, t1))
            if v >= best_iou:
                best, best_iou = (d, text), v
    return best


@dataclass(frozen=True)
class Judgement:
    status: str  # supported | contradicted | unknown
    support: tuple[str, ...] = ()
    contra: tuple[str, ...] = ()
    detail: str = ""


def judge(el: Element, docs: Sequence[EvidenceDoc]) -> Judgement:
    """Compare an element's value with the evidence (exact match after normalisation)."""
    if el.source == "speech":
        hit = aligned_quote(el.span, docs) if el.span else None
        if hit is None:
            return Judgement("unknown", detail="no time-aligned evidence quote")
        doc, text = hit
        if norm(text) == norm(el.value):
            return Judgement("supported", (doc.id,), detail="speech matches aligned quote")
        return Judgement("contradicted", (), (doc.id,), f'aligned quote says: "{text}"')
    facts = [(d, d.facts[el.slot]) for d in docs if el.slot in d.facts]
    if not facts:
        return Judgement("unknown", detail=f"no evidence for {el.slot}")
    sup = tuple(d.id for d, v in facts if norm(v) == norm(el.value))
    con = tuple(d.id for d, v in facts if norm(v) != norm(el.value))
    if len(sup) >= len(con):
        return Judgement("supported", sup, con, f"{el.slot} agrees with evidence")
    return Judgement("contradicted", sup, con,
                     f"evidence says {el.slot}={facts[0][1]!r}" if len(facts) == 1 else f"{len(con)} of {len(facts)} docs disagree")
