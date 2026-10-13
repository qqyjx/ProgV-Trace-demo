"""End-to-end teacher pipeline: parse claims -> generate program -> type-check -> execute -> intervene."""
from __future__ import annotations

from dataclasses import dataclass, field

from .claims import Case, Element, parse_claims
from .dsl import DSLSyntaxError, parse_program
from .interp import Interpreter, Limits, Trace
from .intervene import VERDICT_THRESHOLD, Score, grounding, p_false
from .llm import LLMClient, MockLLM
from .typecheck import DSLTypeError, typecheck

Edge = tuple[str, str, str]
_FALSE_RELS = ("contradicts", "different_time", "different_location")


@dataclass
class Report:
    """The L-EMMA outputs for one case (toy version)."""
    case_id: str
    program_text: str
    elements: list[Element]
    fallback: bool = False
    errors: list[str] = field(default_factory=list)
    p_false: float = 0.0
    verdict: str = "unknown"  # false | true | unknown (fallback)
    label: str = "unknown"  # authentic | forged_caption | altered_speech | mixed
    localized: list[Element] = field(default_factory=list)
    chain: list[Edge] = field(default_factory=list)
    responsibility: dict[str, Score] = field(default_factory=dict)
    trace: Trace | None = None

    @property
    def grounded_ids(self) -> frozenset[str]:
        return frozenset(e.id for e in self.localized)


def provenance_chain(trace: Trace, element_id: str) -> list[Edge]:
    """Evidence chain C for one element: contradicting links, their evidence's source-of links, recontextualisation."""
    g = trace.base.graph
    chain = [e for e in g.incident(element_id, _FALSE_RELS)]
    for ev in sorted({n for e in chain for n in (e[0], e[2]) if g.nodes[n].kind == "evidence"}):
        chain += [e for e in g.incident(ev, ("source_of",)) if e not in chain]
    chain += [e for e in g.incident(element_id, ("recontextualizes",)) if e not in chain]
    return chain


def manipulation_label(localized: list[Element]) -> str:
    srcs = {e.source for e in localized}
    if not srcs:
        return "authentic"
    if srcs <= {"title", "ocr"}:
        return "forged_caption"
    return "altered_speech" if srcs == {"speech"} else "mixed"


def analyze(case: Case, llm: LLMClient | None = None, limits: Limits = Limits()) -> Report:
    """Run the teacher on one case. An ill-typed program triggers the fallback (no verdict)."""
    elements = parse_claims(case)
    text = (llm or MockLLM()).generate_program(elements)
    rep = Report(case.id, text, elements)
    try:
        prog = typecheck(parse_program(text), {e.id: e for e in elements}, limits.max_evidence)
    except DSLTypeError as exc:
        rep.fallback, rep.errors = True, exc.errors
        return rep
    except DSLSyntaxError as exc:
        rep.fallback, rep.errors = True, [str(exc)]
        return rep
    trace = Interpreter(case, limits).run(prog)
    rep.trace = trace
    rep.p_false = p_false(trace.base)
    rep.verdict = "false" if rep.p_false >= VERDICT_THRESHOLD else "true"
    ids = grounding(trace.base)
    rep.localized = [e for e in elements if e.id in ids]
    rep.label = manipulation_label(rep.localized) if rep.verdict == "false" else "authentic"
    for e in rep.localized:
        rep.chain += provenance_chain(trace, e.id)
    rep.responsibility = dict(trace.scores)
    return rep
