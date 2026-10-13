import pytest

from progv.claims import parse_claims
from progv.dsl import parse_program
from progv.evidence import aligned_quote, judge, retrieve
from progv.interp import Interpreter, Limits
from progv.llm import MockLLM
from progv.pipeline import analyze
from progv.synth import generate_cases
from progv.typecheck import typecheck


def run(case, text, limits=Limits()):
    els = {e.id: e for e in parse_claims(case)}
    return Interpreter(case, limits).run(typecheck(parse_program(text), els))


def test_parser_extracts_slots(cases):
    els = {e.id: e for e in parse_claims(cases["toy-forged-caption"])}
    assert els["title.location"].value == "Marlow City"
    assert els["title.time"].value == "2023-03-14"
    assert els["ocr.0"].value == "Port Alder"
    case = cases["toy-forged-caption"]
    s, e = els["title.location"].chars
    assert case.title[s:e] == "Marlow City"


def test_retrieval_by_handle_and_by_words(cases):
    docs = cases["toy-forged-caption"].evidence
    assert [d.id for d in retrieve(docs, "reverse_image", "frame:f1")] == ["ev1"]
    assert retrieve(docs, "reverse_image", "frame:f9") == []
    assert [d.id for d in retrieve(docs, "reverse_image", "harbour road flooding")] == ["ev1"]
    assert retrieve(docs, "news", "frame:f1") == []


def test_temporal_alignment_picks_overlapping_quote(cases):
    docs = cases["toy-altered-speech"].evidence
    assert aligned_quote((7.0, 13.0), docs)[1] == "We will not open the dam gates tonight."
    assert aligned_quote((30.0, 35.0), docs) is None


def test_speech_without_aligned_quote_is_unknown(cases):
    els = {e.id: e for e in parse_claims(cases["toy-altered-speech"])}
    shifted = type(els["speech.s2"])("speech.x", "speech", "event", "x", (50.0, 55.0))
    assert judge(shifted, cases["toy-altered-speech"].evidence).status == "unknown"


def test_forged_caption_graph(cases):
    case = cases["toy-forged-caption"]
    g = analyze(case).trace.base.graph
    assert ("title.location", "different_location", "ev1") in g.edges
    assert ("ev1", "source_of", "frame.f1") in g.edges
    assert ("title.location", "recontextualizes", "frame.f1") in g.edges
    assert ("ev1", "supports", "title.time") in g.edges
    assert ("frame.f1", "same_event", "title.event") in g.edges  # same event, shared across modalities
    assert ("frame.f1", "same_event", "speech.s1") in g.edges  # temporal alignment of speech and footage


def test_authentic_case_has_no_false_relations(cases):
    g = analyze(cases["toy-authentic-control"]).trace.base.graph
    assert not {r for _, r, _ in g.edges} & {"contradicts", "different_time", "different_location", "recontextualizes"}


def test_noisy_or_values(cases):
    rep = analyze(cases["toy-forged-caption"])
    assert rep.p_false == pytest.approx(1 - 0.95 * 0.1)
    assert analyze(cases["toy-mixed"]).p_false == pytest.approx(1 - 0.95 * 0.1 * 0.1)
    assert analyze(cases["toy-authentic-control"]).p_false == pytest.approx(0.05)


def test_evidence_budget_stops_retrieval(cases):
    case = cases["toy-forged-caption"]
    st = run(case, MockLLM().generate_program(parse_claims(case)), Limits(max_evidence=1)).base
    assert any("budget" in n for n in st.notes)
    assert sum(len(v) for v in st.env.values()) == 1


def test_teacher_matches_ground_truth_on_toy_and_generated_cases(cases):
    for case in list(cases.values()) + generate_cases(80, seed=3):
        rep = analyze(case)
        truth = set(case.meta["manipulation"])
        assert rep.grounded_ids == truth, case.id
        assert (rep.verdict == "false") == bool(truth), case.id


@pytest.mark.parametrize("cid, label", [("toy-forged-caption", "forged_caption"), ("toy-altered-speech", "altered_speech"),
                                         ("toy-mixed", "mixed"), ("toy-authentic-control", "authentic")])
def test_labels(cases, cid, label):
    assert analyze(cases[cid]).label == label


def test_ill_typed_llm_output_triggers_fallback(cases):
    class BadLLM:
        def generate_program(self, elements):
            return 'a = ASSIGN(element=title.location)\n'

    rep = analyze(cases["toy-forged-caption"], BadLLM())
    assert rep.fallback and rep.verdict == "unknown" and rep.trace is None
    assert "ASSIGN before any SEARCH" in rep.errors[0]


def test_analysis_is_deterministic(cases):
    a, b = analyze(cases["toy-mixed"]), analyze(cases["toy-mixed"])
    assert a.program_text == b.program_text
    assert {k: v.value for k, v in a.responsibility.items()} == {k: v.value for k, v in b.responsibility.items()}
    assert a.chain == b.chain
