import pytest

from progv.claims import Case, load_all_cases, parse_claims


@pytest.fixture(scope="session")
def cases() -> dict[str, Case]:
    return {c.id: c for c in load_all_cases()}


@pytest.fixture
def elements(cases):
    return lambda cid: {e.id: e for e in parse_claims(cases[cid])}
