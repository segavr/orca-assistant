import pytest
from pydantic import ValidationError

from orca_assistant import CalcSpec, Method, Molecule, ScanSpec, Task, render_input, lint

WATER = "O 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469"


def _spec(**kw):
    return CalcSpec(molecule=Molecule.from_xyz(WATER), **kw)


def test_opt_freq_header():
    text = render_input(_spec())
    assert text.splitlines()[0] == "! B3LYP def2-SVP D3BJ def2/J RIJCOSX TightSCF Opt Freq"
    assert "* xyz 0 1" in text


def test_parity_error():
    with pytest.raises(ValidationError):
        Molecule.from_xyz(WATER, multiplicity=2)


def test_scan_block():
    s = _spec(task=Task.SCAN, scan=ScanSpec(kind="B", atoms=[0, 1], start=0.8, end=1.4, steps=7))
    assert "B 0 1 = 0.8, 1.4, 7" in render_input(s)


def test_scan_requires_spec():
    with pytest.raises(ValidationError):
        _spec(task=Task.SCAN)


def test_lint_double_dispersion():
    s = _spec(method=Method(functional="wB97X-D3", basis="def2-TZVP", dispersion="D3BJ"))
    assert any("double-count" in i.message for i in lint(s))


def test_dispersion_is_not_written_twice():
    """A functional that carries D3 already: the extra field must not reach ORCA."""
    s = _spec(method=Method(functional="wB97X-D3", basis="def2-TZVP", dispersion="D3BJ"))
    tokens = render_input(s).splitlines()[0].split()
    assert "wB97X-D3" in tokens
    assert "D3BJ" not in tokens


def test_plain_functional_keeps_its_dispersion():
    s = _spec(method=Method(functional="B3LYP", basis="def2-TZVP", dispersion="D4"))
    assert "! B3LYP def2-TZVP D4" in render_input(s).splitlines()[0]
