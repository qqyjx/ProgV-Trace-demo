"""Type checker for DSL programs: signatures, enum values, bindings and preconditions."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .claims import Element
from .dsl import DIM_SLOT, ENUMS, OPS, Param, Program


class DSLTypeError(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class CheckedStmt:
    var: str
    op: str
    args: dict[str, tuple[str, str]]  # name -> (kind, payload); kind: enum | element | nodes | str
    line: int


@dataclass(frozen=True)
class CheckedProgram:
    stmts: tuple[CheckedStmt, ...]


def _resolve(p: Param, kind: str, val: str, env: Mapping[str, str],
             elements: Mapping[str, Element]) -> tuple[str, str] | None:
    if kind == "str":
        return ("str", val) if "Str" in p.accepts else None
    for a in p.accepts:
        if a in ENUMS and val in ENUMS[a]:
            return ("enum", val)
        if a == "Element" and val in elements:
            return ("element", val)
        if a == "Nodes" and env.get(val) == "Nodes":
            return ("nodes", val)
    return None


def typecheck(program: Program, elements: Mapping[str, Element], max_search: int = 10) -> CheckedProgram:
    """Return a CheckedProgram or raise DSLTypeError listing every problem found."""
    errors: list[str] = []
    env: dict[str, str] = {}  # variable -> result type
    intervened: set[str] = set()
    n_search = 0
    checked: list[CheckedStmt] = []
    for st in program.stmts:
        def err(msg: str, st=st) -> None:
            errors.append(f"line {st.line}: {msg}")

        spec = OPS.get(st.op)
        if spec is None:
            err(f"unknown operator {st.op}")
            continue
        if st.var in env:
            err(f"variable {st.var!r} is already bound")
        given: dict[str, tuple[str, str]] = {}
        for name, kind, val in st.args:
            if name in given:
                err(f"{st.op}: duplicate argument {name!r}")
            given[name] = (kind, val)
        names = {p.name for p in spec.params}
        for name in given:
            if name not in names:
                err(f"{st.op}: unknown argument {name!r}")
        resolved: dict[str, tuple[str, str]] = {}
        for p in spec.params:
            if p.name not in given:
                err(f"{st.op}: missing argument {p.name!r}")
                continue
            kind, val = given[p.name]
            r = _resolve(p, kind, val, env, elements)
            if r is None:
                err(f"{st.op}.{p.name} expects {' or '.join(p.accepts)}, got {val!r}")
            else:
                resolved[p.name] = r
        if len(resolved) == len(spec.params):
            _semantic(st.op, resolved, elements, intervened, n_search, max_search, err)
            if st.op == "SEARCH":
                n_search += 1
            if st.op == "INTERVENE":
                intervened.add(resolved["span"][1])
        env[st.var] = spec.returns
        checked.append(CheckedStmt(st.var, st.op, resolved, st.line))
    if errors:
        raise DSLTypeError(errors)
    return CheckedProgram(tuple(checked))


def _semantic(op, a, elements, intervened, n_search, max_search, err) -> None:
    """Cross-argument rules and preconditions (well-formedness constraints)."""
    if op == "SEARCH" and n_search + 1 > max_search:
        err(f"SEARCH exceeds the evidence budget of {max_search}")
    elif op == "VERIFY":
        el = elements[a["element"][1]]
        if a["attr"][1] != el.slot:
            err(f"VERIFY attr {a['attr'][1]} does not match slot {el.slot} of {el.id}")
    elif op == "CHECK":
        el = elements[a["claimed"][1]]
        if DIM_SLOT[a["dim"][1]] != el.slot:
            err(f"CHECK dim {a['dim'][1]} cannot compare slot {el.slot} of {el.id}")
    elif op == "INTERVENE":
        el = elements[a["span"][1]]
        act = a["action"][1]
        if act == "rewrite" and el.source != "speech":
            err(f"INTERVENE rewrite applies to speech segments, not {el.id}")
        if act == "replace" and el.source == "speech":
            err(f"INTERVENE replace does not apply to speech segment {el.id}; use rewrite")
    elif op == "ASSIGN":
        eid = a["element"][1]
        if n_search == 0:
            err("ASSIGN before any SEARCH")
        if eid not in intervened:
            err(f"ASSIGN({eid}) needs an earlier INTERVENE on the same element")
