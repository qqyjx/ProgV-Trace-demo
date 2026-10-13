"""Toy cases and Module A (claim parser): a rule-based parser that builds claim elements U."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCATIONS = ("Port Alder", "Marlow City", "Kestrel Bay", "Dunmere",
             "Hollin Ridge", "Saltwick", "Ormond Falls", "Veyra")
EVENT_LEXICON = {
    "flood": ("flood", "floods", "flooding"),
    "fire": ("fire", "fires", "blaze"),
    "protest": ("protest", "protests", "march"),
    "press conference": ("press conference",),
    "storm": ("storm", "storms"),
}
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DATA_DIR = Path(__file__).parent / "data" / "cases"


@dataclass(frozen=True)
class Element:
    """One claim element u_i. `slot` is a DSL SlotType; `span` is seconds, `chars` an offset in the title."""
    id: str
    source: str  # title | ocr | frame | speech
    slot: str
    value: str
    span: tuple[float, float] | None = None
    chars: tuple[int, int] | None = None


@dataclass(frozen=True)
class EvidenceDoc:
    id: str
    type: str  # news | fact_check | reverse_image | knowledge_base
    source: str
    queries: tuple[str, ...]
    text: str
    facts: dict[str, str] = field(default_factory=dict)
    quotes: tuple[tuple[float, float, str], ...] = ()


@dataclass(frozen=True)
class Case:
    """A synthetic toy post: title, OCR, keyframe tags, ASR segments and an evidence store."""
    id: str
    title: str
    ocr: tuple[dict[str, Any], ...]
    frames: tuple[dict[str, Any], ...]
    speech: tuple[dict[str, Any], ...]
    evidence: tuple[EvidenceDoc, ...]
    meta: dict[str, Any] = field(default_factory=dict)  # ground truth for tests; never read by the LLM or interpreter

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Case":
        if d.get("synthetic") is not True:
            raise ValueError("toy cases must carry \"synthetic\": true")
        docs = tuple(EvidenceDoc(
            id=e["id"], type=e["type"], source=e["source"], queries=tuple(e["queries"]),
            text=e["text"], facts=dict(e.get("facts", {})),
            quotes=tuple((q["t0"], q["t1"], q["text"]) for q in e.get("quotes", [])),
        ) for e in d["evidence"])
        return Case(d["id"], d["title"], tuple(d.get("ocr", [])), tuple(d["frames"]),
                    tuple(d["speech"]), docs, dict(d.get("meta", {})))


def load_case(path: str | Path) -> Case:
    return Case.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_all_cases(directory: Path = DATA_DIR) -> list[Case]:
    return [load_case(p) for p in sorted(directory.glob("*.json"))]


def find_location(text: str) -> tuple[str, int, int] | None:
    for loc in LOCATIONS:
        m = re.search(re.escape(loc), text, re.IGNORECASE)
        if m:
            return loc, m.start(), m.end()
    return None


def find_event(text: str) -> tuple[str, int, int] | None:
    best = None
    for canon, aliases in EVENT_LEXICON.items():
        for a in aliases:
            m = re.search(r"\b" + re.escape(a) + r"\b", text, re.IGNORECASE)
            if m and (best is None or m.start() < best[1]):
                best = (canon, m.start(), m.end())
    return best


def parse_claims(case: Case) -> list[Element]:
    """Decompose the post into claim elements U (title slots, OCR, keyframes, speech segments)."""
    els: list[Element] = []
    if (ev := find_event(case.title)):
        els.append(Element("title.event", "title", "event", ev[0], None, (ev[1], ev[2])))
    if (loc := find_location(case.title)):
        els.append(Element("title.location", "title", "location", loc[0], None, (loc[1], loc[2])))
    if (m := DATE_RE.search(case.title)):
        els.append(Element("title.time", "title", "time", m.group(0), None, (m.start(), m.end())))
    for i, line in enumerate(case.ocr):
        if (loc := find_location(line["text"])):
            els.append(Element(f"ocr.{i}", "ocr", "location", loc[0], (line["t0"], line["t1"])))
    for f in case.frames:
        els.append(Element(f"frame.{f['id']}", "frame", "event", f["visual"]["event"], (f["t0"], f["t1"])))
    for s in case.speech:
        els.append(Element(f"speech.{s['id']}", "speech", "event", s["text"], (s["t0"], s["t1"])))
    return els
