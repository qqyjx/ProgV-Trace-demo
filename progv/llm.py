"""Module B (toy): the verification-program generator behind an `LLMClient` interface.

`MockLLM` is deterministic and offline. `RemoteLLM` is an optional hook for a real model; it is off by
default, needs the provider SDK plus credentials in the environment, and is not exercised by the tests.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Protocol

from .claims import Element
from .dsl import OPS


class LLMClient(Protocol):
    def generate_program(self, elements: Sequence[Element]) -> str:
        """Return DSL text for the given claim elements."""


def _v(prefix: str, element_id: str) -> str:
    return f"{prefix}_{element_id.replace('.', '_')}"


class MockLLM:
    """Template planner: one retrieval per evidence type, one check per element, then expand and intervene."""

    def generate_program(self, elements: Sequence[Element]) -> str:
        frames = [e for e in elements if e.source == "frame"]
        speech = [e for e in elements if e.source == "speech"]
        lines = ["# 1. schedule evidence retrieval"]
        img = sp = None
        if frames:
            img = "img"
            lines.append(f'img = SEARCH(type=reverse_image, query="frame:{frames[0].id.split(".", 1)[1]}")')
        fc = "fc"
        lines.append('fc = SEARCH(type=fact_check, query="title")')
        if speech:
            sp = "sp"
            lines.append('sp = SEARCH(type=news, query="speech")')
        ref = img or fc
        lines.append("# 2. decompose the claim into sub-claims and verify each one")
        plan: list[tuple[Element, str]] = []  # (element, evidence variable used for interventions)
        for e in elements:
            if e.source == "speech":
                lines.append(f"{_v('v', e.id)} = VERIFY(attr=event, element={e.id}, evidence={sp})")
                plan.append((e, sp))
            elif e.slot == "event" and e.source == "title":
                lines.append(f"{_v('v', e.id)} = VERIFY(attr=event, element={e.id}, evidence={fc})")
                plan.append((e, fc))
            elif e.source == "frame":
                lines.append(f"{_v('v', e.id)} = VERIFY(attr=event, element={e.id}, evidence={ref})")
                plan.append((e, ref))
            else:
                dim = "temporal" if e.slot == "time" else "location"
                lines.append(f"{_v('c', e.id)} = CHECK(dim={dim}, claimed={e.id}, evidence={ref})")
                plan.append((e, ref))
        lines.append("# 3. expand the provenance graph")
        if img:
            lines.append(f"x_src = EXPAND(node={img}, relation=source_of)")
            lines.append(f"x_ev = EXPAND(node={frames[0].id}, relation=same_event)")
            lines.append(f"x_ctx = EXPAND(node={frames[0].id}, relation=recontextualizes)")
        lines.append("# 4. plan minimal interventions and assign responsibility")
        for e, ev in plan:
            action = "rewrite" if e.source == "speech" else "replace"
            lines.append(f"{_v('i', e.id)} = INTERVENE(span={e.id}, action={action}, target={ev})")
            lines.append(f"{_v('r', e.id)} = ASSIGN(element={e.id})")
        return "\n".join(lines) + "\n"


def dsl_spec() -> str:
    """Operator signatures, as shown to a real model in the prompt."""
    lines = []
    for o in OPS.values():
        params = ", ".join(p.name + ":" + "|".join(p.accepts) for p in o.params)
        lines.append(f"{o.name}({params}) -> {o.returns}")
    return "\n".join(lines)


class RemoteLLM:
    """Optional hook (`--llm openai|anthropic`). Set PROGV_LLM_MODEL; the SDK reads its own API-key variable."""

    def __init__(self, provider: str):
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"unknown provider {provider!r}")
        self.provider = provider
        self.model = os.environ.get("PROGV_LLM_MODEL")
        if not self.model:
            raise RuntimeError("set PROGV_LLM_MODEL to use a remote model")

    def generate_program(self, elements: Sequence[Element]) -> str:
        prompt = ("Write a verification program in this typed DSL, one statement per line "
                  "(`var = OP(name=value, ...)`). Output only the program.\n"
                  f"Operators:\n{dsl_spec()}\n"
                  "Claim elements:\n" + json.dumps([e.__dict__ for e in elements], default=list))
        if self.provider == "openai":
            from openai import OpenAI  # type: ignore
            r = OpenAI().chat.completions.create(model=self.model, messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content
        import anthropic  # type: ignore
        r = anthropic.Anthropic().messages.create(model=self.model, max_tokens=2000,
                                                  messages=[{"role": "user", "content": prompt}])
        return r.content[0].text


def make_client(name: str) -> LLMClient:
    return MockLLM() if name == "mock" else RemoteLLM(name)
