# ProgV-Trace demo

A small, offline demo of **ProgV-Trace**, the framework from the paper
*"ProgV-Trace: An LLM-Driven Verification Framework for Evidence-Grounded Video Misinformation Attribution"*
(Yifan Xiao, Shijie Li, Yu Huang; ACM Multimedia 2026).

> **This is a minimal demo that illustrates the method of the paper. It is not the code used for the paper's
> experiments, and it does not reproduce any number reported in the paper.** All inputs are synthetic toy data.
> The demo prints only values that it computes on those toy inputs.

## The problem and the method

Video misinformation is often assembled from real footage, a forged caption and altered speech. A binary
detector does not say which element makes the narrative false, or by how much. The paper defines the task
*evidence-grounded multimedia misinformation attribution*. A system must output:

1. a veracity verdict,
2. a fine-grained localization of the false elements,
3. an evidence provenance chain,
4. a responsibility score for each element.

ProgV-Trace lets an LLM write an executable *verification program* in a typed domain-specific language (DSL).
The program decomposes the claim, schedules evidence retrieval, and expands a temporal multimodal provenance
graph. The framework then plans minimal counterfactual interventions: it replaces one element with its
evidence-consistent counterpart and measures how the outcome changes. This gives each element a responsibility
score. A distillation stage transfers the programmatic reasoning into a small verifier that needs no LLM at
deployment.

## Quickstart

```bash
pip install -e ".[test]"      # numpy; pytest for the tests
python -m progv.demo          # about one second, no network, no API key
pytest -q
```

Options: `--case <id>`, `--show-program` (print the program of every case), `--train-cases N`, and
`--llm openai|anthropic`. The last option is a hook for a real model. It is off by default, needs the provider
SDK, `PROGV_LLM_MODEL` and credentials in the environment, and is not tested here.

## What the demo shows

For each case, `python -m progv.demo` prints the claim elements, the generated DSL program, the type-check result,
the sub-claim verdicts, the verdict with its type, the localized false elements, the provenance chain, the
responsibility score of every element, and the verdict of the distilled student.

| Paper component | Demo module | What it does here |
|---|---|---|
| Multimodal Claim Parser (Module A) | `progv/claims.py` | Rule-based extraction of claim elements: title slots, OCR, keyframe tags, speech segments |
| Verification Program Generator (Module B) | `progv/llm.py`, `progv/dsl.py`, `progv/typecheck.py` | `LLMClient` interface, deterministic `MockLLM`, the six typed operators, a text parser and a type checker |
| Program-Guided Retrieval and Graph Expansion (Module C) | `progv/evidence.py`, `progv/graph.py`, `progv/interp.py` | `SEARCH` over a toy evidence store; `EXPAND` adds typed edges to a provenance graph |
| Program Executor and Verdict Aggregator (Module D) | `progv/interp.py`, `progv/intervene.py` | Sub-claim verdicts from `VERIFY`/`CHECK`, noisy-OR aggregation |
| Minimal Counterfactual Intervention (Module E) | `progv/intervene.py`, `progv/interp.py` | `INTERVENE` plans a repair; `ASSIGN` re-runs the program and scores R(u) |
| Program-to-Student Distillation (Module F) | `progv/student.py` | Two logistic heads (numpy) trained on teacher outputs; labelled toy |

### The DSL

The six operators follow the paper. Names in `Nodes` position are variables bound by `SEARCH`.

```
SEARCH(type:EvidenceType, query:Str)                          -> Nodes
VERIFY(attr:SlotType, element:Element, evidence:Nodes)        -> Tri      # supported / contradicted / unknown
CHECK(dim:Dimension, claimed:Element, evidence:Nodes)         -> Match    # dim in {temporal, location}
EXPAND(node:Nodes|Element, relation:EdgeType)                 -> Graph
INTERVENE(span:Element, action:Action, target:Str|Nodes)      -> Delta    # replace | remove | rewrite
ASSIGN(element:Element)                                       -> Score
```

The type checker rejects, among others: unknown operators and enum values, missing or duplicate arguments,
unbound variables, an `attr` that does not match the element slot, `rewrite` on a non-speech element,
`ASSIGN` before any `SEARCH`, `ASSIGN` without an earlier `INTERVENE` on the same element, and more than K=10
`SEARCH` statements. An ill-typed program makes the pipeline fall back and return no verdict.

Claim decomposition is the parser step (Module A). Temporal alignment appears in two places: speech segments are
checked against the evidence quote with the largest time overlap, and `EXPAND(..., same_event)` links a speech
segment to the video segment it overlaps.

### Toy data

`progv/data/cases/` holds four hand-written synthetic posts (fictional places, no real people or footage):

| Case | Construction | Expected false element |
|---|---|---|
| `toy-forged-caption` | real harbour flood footage, caption names another city | `title.location` |
| `toy-altered-speech` | real press conference, one speech segment negated | `speech.s2` |
| `toy-mixed` | real fire footage, forged date and altered speech | `title.time`, `speech.s2` |
| `toy-authentic-control` | consistent post | none |

`progv/synth.py` generates more posts of the same schema from seeded templates. They train and test the toy student.

## Repository layout

```
progv/            package: claims, dsl, typecheck, evidence, graph, interp, intervene, llm, pipeline, student, synth, demo
progv/data/cases/ four synthetic example posts (JSON)
tests/            pytest: type checker, interpreter, interventions, student, demo
```

## Simplifications

This demo differs from the paper in these ways.

- **Perception and learned models.** There are no video, speech or OCR models, no BERT or CLIP encoders, no R-GCN, and no
  attention pooling. Claim parsing uses a small lexicon and regular expressions. Relation classifiers are rules.
- **Program generator.** `MockLLM` is a fixed template. It sees only the parsed claim elements. It plans an
  intervention for every element, not only for elements above a grounding threshold.
- **Operator signatures.** `VERIFY`, `CHECK` and `INTERVENE` take claim-element IDs and an evidence variable, not raw strings.
  `INTERVENE` with an evidence variable resolves to the evidence-consistent counterpart at run time.
  `CHECK` supports two dimensions only. Edge names use underscores (`source_of`).
- **Provenance graph.** Three node kinds (text span, video segment, evidence) and the seven edge types. The paper also has entity and anchor nodes.
- **Executor and verdict.** Sub-claims are judged by exact match after normalization, against retrieved evidence. The verdict is a
  noisy-OR over per-element statuses with fixed strengths, not an LLM controller with uncertainty. The paper's fallback to direct
  chain-of-thought reasoning is replaced by "no verdict".
- **Responsibility.** R(u) = 0.4 d_verdict + 0.3 d_path + 0.2 d_ground + 0.1 d_conf uses the paper's weights. The terms are defined here by this demo:
  d_verdict is the change in P(false), d_path is the Jaccard distance between the edge sets of the original and the repaired run,
  d_ground is the Jaccard distance between the grounded element sets, and d_conf is the change in |2P-1|. The repaired run re-executes the
  whole program; there is no L-hop local re-computation. Each element is repaired alone, so when two elements are false
  (`toy-mixed`), repairing one leaves the verdict false and each R(u) is small. Elements that already agree with the evidence get a no-op
  intervention and R = 0.
- **Student.** Two logistic regressions on 13 hand-made features plus a bias, trained with soft cross-entropy against the teacher's R(u) and with the
  teacher's verdict. The paper uses a BERT-base student with a KL distillation loss and further supervision signals. The toy student still reads
  the case's evidence store; it does not retrieve. The agreement it prints is between the student and the teacher on templated synthetic cases.
  It measures nothing about real videos.
- **Not included.** The Program-Hard Challenge Protocol, the training objectives, all datasets, and every experiment of the paper.

## Citation

```bibtex
@inproceedings{xiao2026progvtrace,
  author    = {Xiao, Yifan and Li, Shijie and Huang, Yu},
  title     = {{ProgV-Trace}: An {LLM}-Driven Verification Framework for Evidence-Grounded Video Misinformation Attribution},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia (MM '26)},
  year      = {2026},
  address   = {Rio de Janeiro, Brazil},
  publisher = {ACM}
}
```

The DOI will be added once it is registered.

## License

MIT. See `LICENSE`.
