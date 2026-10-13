"""Temporal-multimodal provenance graph (text spans, video segments, evidence nodes; the paper's seven edge types)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    kind: str  # text_span | video_segment | evidence
    attrs: dict[str, Any] = field(default_factory=dict)


class ProvGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: set[tuple[str, str, str]] = set()  # (src, relation, dst)

    def add_node(self, node: Node) -> None:
        self.nodes.setdefault(node.id, node)

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        self.edges.add((src, rel, dst))

    def incident(self, node_id: str, rels: tuple[str, ...] | None = None) -> list[tuple[str, str, str]]:
        return sorted(e for e in self.edges
                      if node_id in (e[0], e[2]) and (rels is None or e[1] in rels))


def jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)
