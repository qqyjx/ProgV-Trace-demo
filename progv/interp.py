"""Interpreter for checked programs: retrieval, graph expansion, sub-claim verdicts, interventions."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .claims import Case, Element, EvidenceDoc, parse_claims
from .evidence import Judgement, iou, judge, norm, retrieve
from .graph import Node, ProvGraph
from .intervene import Delta, Score, plan_delta, responsibility
from .typecheck import CheckedProgram


@dataclass(frozen=True)
class Limits:
    """STOP conditions: evidence budget K and a graph node cap."""
    max_evidence: int = 10
    max_nodes: int = 50


@dataclass
class SubClaim:
    element: str
    op: str
    judgement: Judgement


@dataclass
class State:
    """Result of one execution pass over the verification operators."""
    graph: ProvGraph
    elements: dict[str, Element]
    subclaims: list[SubClaim]
    env: dict[str, list[EvidenceDoc]]
    notes: list[str] = field(default_factory=list)

    def element_status(self) -> dict[str, str]:
        """Per-element status; contradicted beats supported beats unknown."""
        rank = {"unknown": 0, "supported": 1, "contradicted": 2}
        out: dict[str, str] = {}
        for sc in self.subclaims:
            s = sc.judgement.status
            if sc.element not in out or rank[s] > rank[out[sc.element]]:
                out[sc.element] = s
        return out


@dataclass
class Trace:
    base: State
    deltas: dict[str, Delta]
    scores: dict[str, Score]


def _event_of(n: Node) -> str | None:
    if n.kind == "evidence":
        v = n.attrs["doc"].facts.get("event")
    else:
        el = n.attrs["element"]
        v = el.value if el.slot == "event" and el.source != "speech" else None
    return norm(v) if v else None


def _relation(rel: str, a: Node, b: Node, g: ProvGraph) -> bool:
    """Rule-based stand-in for the relation classifiers: does `rel` hold for the ordered pair (a, b)?"""
    if rel in ("supports", "contradicts", "different_time", "different_location"):
        if a.kind != "evidence" or b.kind == "evidence":
            return False
        want = {"supports": "supported", "contradicts": "contradicted",
                "different_time": "contradicted", "different_location": "contradicted"}[rel]
        el = b.attrs["element"]
        if (rel == "different_time" and el.slot != "time") or (rel == "different_location" and el.slot != "location"):
            return False
        return judge(el, [a.attrs["doc"]]).status == want
    if rel == "source_of":
        return (a.kind == "evidence" and a.attrs["doc"].type == "reverse_image" and b.kind == "video_segment"
                and f"frame:{b.attrs['element'].id.split('.', 1)[1]}" in a.attrs["doc"].queries)
    if rel == "same_event":
        if a.id >= b.id:
            return False
        ea, eb = _event_of(a), _event_of(b)
        if ea and eb:
            return ea == eb
        # temporal alignment: a speech segment overlapping a video segment
        pair = [x for x in (a, b) if x.kind != "evidence"]
        if len(pair) == 2 and {x.kind for x in pair} == {"text_span", "video_segment"}:
            sp = next(x for x in pair if x.kind == "text_span").attrs["element"]
            vid = next(x for x in pair if x.kind == "video_segment").attrs["element"]
            return sp.source == "speech" and iou(sp.span, vid.span) > 0
        return False
    if rel == "recontextualizes":
        if a.kind != "text_span" or b.kind != "video_segment" or a.attrs["element"].slot != "location":
            return False
        origins = [s for (s, r, d) in g.edges if r == "source_of" and d == b.id]
        locs = {norm(g.nodes[s].attrs["doc"].facts["location"]) for s in origins
                if "location" in g.nodes[s].attrs["doc"].facts}
        return bool(locs) and norm(a.attrs["element"].value) not in locs
    return False


class Interpreter:
    def __init__(self, case: Case, limits: Limits = Limits()):
        self.case, self.limits = case, limits
        self.elements = parse_claims(case)

    def run(self, prog: CheckedProgram) -> Trace:
        """Run the verification pass, then plan each INTERVENE and re-run the pass for each ASSIGN."""
        base = self._pass(prog, {})
        deltas: dict[str, Delta] = {}
        scores: dict[str, Score] = {}
        by_id = {e.id: e for e in self.elements}
        for st in prog.stmts:
            if st.op == "INTERVENE":
                eid = st.args["span"][1]
                deltas[eid] = plan_delta(by_id[eid], st.args["action"][1], st.args["target"], base)
            elif st.op == "ASSIGN":
                eid = st.args["element"][1]
                pert = self._pass(prog, deltas[eid].overrides())
                scores[eid] = responsibility(base, pert, eid)
        return Trace(base, deltas, scores)

    def _pass(self, prog: CheckedProgram, overrides: dict[str, str | None]) -> State:
        els: dict[str, Element] = {}
        for el in self.elements:
            if el.id in overrides:
                if overrides[el.id] is None:
                    continue
                el = replace(el, value=overrides[el.id])
            els[el.id] = el
        g = ProvGraph()
        for el in els.values():
            g.add_node(Node(el.id, "video_segment" if el.source == "frame" else "text_span", {"element": el}))
        st = State(g, els, [], {})
        retrieved = 0
        for s in prog.stmts:
            a = s.args
            if s.op == "SEARCH":
                docs = retrieve(self.case.evidence, a["type"][1], a["query"][1])
                room = max(self.limits.max_evidence - retrieved, 0)
                if len(docs) > room:
                    st.notes.append(f"evidence budget K={self.limits.max_evidence} exhausted at {s.var}")
                    docs = docs[:room]
                retrieved += len(docs)
                for d in docs:
                    if len(g.nodes) < self.limits.max_nodes:
                        g.add_node(Node(d.id, "evidence", {"doc": d}))
                st.env[s.var] = docs
            elif s.op in ("VERIFY", "CHECK"):
                key = "element" if s.op == "VERIFY" else "claimed"
                el = els.get(a[key][1])
                if el is None:  # removed by an intervention
                    continue
                j = judge(el, st.env[a["evidence"][1]])
                st.subclaims.append(SubClaim(el.id, s.op, j))
                bad = {"time": "different_time"}.get(el.slot, "different_location")
                for d in j.support:
                    g.add_edge(d, "supports", el.id)
                for d in j.contra:
                    if s.op == "VERIFY":
                        g.add_edge(d, "contradicts", el.id)
                    else:
                        g.add_edge(el.id, bad, d)
            elif s.op == "EXPAND":
                kind, ref = a["node"]
                focus = [d.id for d in st.env[ref]] if kind == "nodes" else ([ref] if ref in g.nodes else [])
                for f in focus:
                    for o in sorted(g.nodes):
                        if o == f:
                            continue
                        for x, y in ((f, o), (o, f)):
                            if _relation(a["relation"][1], g.nodes[x], g.nodes[y], g):
                                g.add_edge(x, a["relation"][1], y)
        return st
