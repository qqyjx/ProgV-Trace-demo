import pytest

from progv import demo


def test_demo_prints_all_sections(capsys):
    demo.main([])
    out = capsys.readouterr().out
    for needle in ("SYNTHETIC", "verdict: FALSE", "localized false element(s):", "provenance chain",
                   "responsibility R(u)", "distilled student", "verdict: TRUE", "held-out generated"):
        assert needle in out


def test_demo_single_case(capsys):
    demo.main(["--case", "toy-altered-speech"])
    out = capsys.readouterr().out
    assert "toy-altered-speech" in out and "toy-mixed" not in out


def test_unknown_case_exits():
    with pytest.raises(SystemExit):
        demo.main(["--case", "nope"])


def test_demo_output_is_deterministic(capsys):
    demo.main([])
    a = capsys.readouterr().out
    demo.main([])
    assert capsys.readouterr().out == a
