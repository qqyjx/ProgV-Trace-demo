import pytest

from progv.claims import load_all_cases, parse_claims
from progv.dsl import DSLSyntaxError, parse_program
from progv.llm import MockLLM
from progv.typecheck import DSLTypeError, typecheck

BASE = '''
img = SEARCH(type=reverse_image, query="frame:f1")
c = CHECK(dim=location, claimed=title.location, evidence=img)
'''


def check(text, cid="toy-forged-caption", **kw):
    case = next(c for c in load_all_cases() if c.id == cid)
    return typecheck(parse_program(text), {e.id: e for e in parse_claims(case)}, **kw)


@pytest.mark.parametrize("case", load_all_cases(), ids=lambda c: c.id)
def test_mock_programs_are_well_typed(case):
    els = parse_claims(case)
    typecheck(parse_program(MockLLM().generate_program(els)), {e.id: e for e in els})


def test_valid_program_is_resolved():
    prog = check(BASE)
    assert [s.op for s in prog.stmts] == ["SEARCH", "CHECK"]
    assert prog.stmts[1].args["claimed"] == ("element", "title.location")
    assert prog.stmts[1].args["evidence"] == ("nodes", "img")


def test_quoted_commas_survive_parsing():
    prog = parse_program('e = SEARCH(type=news, query="flood, harbour, 2023")')
    assert prog.stmts[0].args[1] == ("query", "str", "flood, harbour, 2023")


@pytest.mark.parametrize("bad, fragment", [
    ('x = FETCH(type=news, query="a")', "unknown operator"),
    ('x = SEARCH(type=video, query="a")', "EvidenceType"),
    ('x = SEARCH(type=news)', "missing argument 'query'"),
    ('x = SEARCH(type=news, query="a", extra=news)', "unknown argument"),
    ('x = SEARCH(type=news, query=news)', "expects Str"),
    ('x = SEARCH(type=news, type=news, query="a")', "duplicate argument"),
    (BASE + 'c = SEARCH(type=news, query="a")', "already bound"),
    (BASE + 'v = VERIFY(attr=event, element=title.event, evidence=nope)', "expects Nodes"),
    (BASE + 'v = VERIFY(attr=event, element=title.event, evidence=c)', "expects Nodes"),
    (BASE + 'v = VERIFY(attr=event, element=title.nothing, evidence=img)', "expects Element"),
    (BASE + 'v = VERIFY(attr=location, element=title.event, evidence=img)', "does not match slot"),
    (BASE + 'v = CHECK(dim=temporal, claimed=title.location, evidence=img)', "cannot compare"),
    (BASE + 'i = INTERVENE(span=title.location, action=rewrite, target=img)', "applies to speech"),
    (BASE + 'i = INTERVENE(span=speech.s1, action=replace, target=img)', "use rewrite"),
    (BASE + 'i = INTERVENE(span=title.location, action=delete, target=img)', "expects Action"),
    ('a = ASSIGN(element=title.location)', "ASSIGN before any SEARCH"),
    (BASE + 'a = ASSIGN(element=title.location)', "earlier INTERVENE"),
])
def test_ill_typed_programs_are_rejected(bad, fragment):
    with pytest.raises(DSLTypeError) as exc:
        check(bad)
    assert fragment in str(exc.value)


def test_all_errors_are_reported_together():
    with pytest.raises(DSLTypeError) as exc:
        check('a = FETCH(x=news)\nb = SEARCH(type=video, query="q")')
    assert len(exc.value.errors) == 2


def test_search_budget_is_enforced():
    prog = "\n".join(f'e{i} = SEARCH(type=news, query="q")' for i in range(3))
    check(prog, max_search=3)
    with pytest.raises(DSLTypeError, match="evidence budget"):
        check(prog, max_search=2)


@pytest.mark.parametrize("bad", ["x SEARCH(type=news)", 'x = SEARCH(type=news, query="open)', "x = SEARCH(news)"])
def test_syntax_errors(bad):
    with pytest.raises(DSLSyntaxError):
        parse_program(bad)
