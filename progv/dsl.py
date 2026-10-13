"""The typed verification DSL: operator signatures, text syntax and parser.

Syntax (one statement per line, `#` starts a comment line):
    ev = SEARCH(type=reverse_image, query="frame:f1")
    c1 = CHECK(dim=location, claimed=title.location, evidence=ev)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SLOT_TYPES = ("entity", "event", "time", "location", "quantity", "relation")
EVIDENCE_TYPES = ("news", "fact_check", "reverse_image", "knowledge_base")
EDGE_TYPES = ("supports", "contradicts", "same_event", "different_time",
              "different_location", "recontextualizes", "source_of")
ACTIONS = ("replace", "remove", "rewrite")
DIMENSIONS = ("temporal", "location")
ENUMS = {"SlotType": SLOT_TYPES, "EvidenceType": EVIDENCE_TYPES, "EdgeType": EDGE_TYPES,
         "Action": ACTIONS, "Dimension": DIMENSIONS}
# Dimension -> element slot it can compare.
DIM_SLOT = {"temporal": "time", "location": "location"}


@dataclass(frozen=True)
class Param:
    name: str
    accepts: tuple[str, ...]  # enum names, "Str", "Element" or "Nodes"


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: tuple[Param, ...]
    returns: str  # Nodes | Tri | Match | Graph | Delta | Score


OPS: dict[str, OpSpec] = {s.name: s for s in (
    OpSpec("SEARCH", (Param("type", ("EvidenceType",)), Param("query", ("Str",))), "Nodes"),
    OpSpec("VERIFY", (Param("attr", ("SlotType",)), Param("element", ("Element",)),
                      Param("evidence", ("Nodes",))), "Tri"),
    OpSpec("CHECK", (Param("dim", ("Dimension",)), Param("claimed", ("Element",)),
                     Param("evidence", ("Nodes",))), "Match"),
    OpSpec("EXPAND", (Param("node", ("Nodes", "Element")), Param("relation", ("EdgeType",))), "Graph"),
    OpSpec("INTERVENE", (Param("span", ("Element",)), Param("action", ("Action",)),
                         Param("target", ("Str", "Nodes"))), "Delta"),
    OpSpec("ASSIGN", (Param("element", ("Element",)),), "Score"),
)}


class DSLSyntaxError(ValueError):
    pass


@dataclass(frozen=True)
class Stmt:
    var: str
    op: str
    args: tuple[tuple[str, str, str], ...]  # (name, "str" | "ident", text)
    line: int


@dataclass(frozen=True)
class Program:
    stmts: tuple[Stmt, ...]


_LINE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*([A-Z]+)\((.*)\)$")
_IDENT = re.compile(r"^[A-Za-z_][\w.]*$")


def _split_args(s: str, line: int) -> list[str]:
    parts, buf, quoted = [], [], False
    for ch in s:
        if ch == '"':
            quoted = not quoted
        if ch == "," and not quoted:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if quoted:
        raise DSLSyntaxError(f"line {line}: unterminated string")
    tail = "".join(buf)
    if tail.strip() or parts:
        parts.append(tail)
    return parts


def parse_program(text: str) -> Program:
    """Parse DSL text into a Program. Raises DSLSyntaxError on malformed lines."""
    stmts = []
    for no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise DSLSyntaxError(f"line {no}: expected `var = OP(name=value, ...)`: {line!r}")
        var, op, body = m.groups()
        args = []
        for piece in _split_args(body, no):
            if "=" not in piece:
                raise DSLSyntaxError(f"line {no}: argument without name: {piece.strip()!r}")
            name, val = (x.strip() for x in piece.split("=", 1))
            if len(val) >= 2 and val[0] == val[-1] == '"':
                args.append((name, "str", val[1:-1]))
            elif _IDENT.match(val):
                args.append((name, "ident", val))
            else:
                raise DSLSyntaxError(f"line {no}: bad value {val!r}")
        stmts.append(Stmt(var, op, tuple(args), no))
    return Program(tuple(stmts))
