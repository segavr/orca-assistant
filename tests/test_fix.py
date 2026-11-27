from pathlib import Path

import pytest

from orca_assistant.fix import make_fix, next_fix_path

DATA = Path(__file__).parent / "data"

EQ = "O 0.000000 0.000000 0.117000\nH 0.000000 0.757000 -0.469000\nH 0.000000 -0.757000 -0.469000"

# The exact inputs that produced the real failing outputs in tests/data/fail_*.out.
SCF_INP = f"! B3LYP def2-SVP\n%maxcore 500\n%scf MaxIter 2 end\n* xyz 0 1\n{EQ}\n*\n"
OPT_INP = ("! B3LYP def2-SVP Opt\n%maxcore 500\n%geom MaxIter 1 end\n* xyz 0 1\n"
           "O 0.0 0.0 0.117\nH 0.0 1.1 -0.6\nH 0.0 -1.1 -0.6\n*\n")
MULT_INP = f"! B3LYP def2-SVP\n%maxcore 500\n* xyz 0 2\n{EQ}\n*\n"
BASIS_INP = f"! B3LYP def2-SVPP\n%maxcore 500\n* xyz 0 1\n{EQ}\n*\n"

SCF_OUT = ("               *        SCF NOT CONVERGED AFTER   1 CYCLES         *\n"
           "ORCA finished by error termination in LEANSCF\n")
OPT_OUT = ("CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
           "  O      0.000000    0.000000    0.120662\n"
           "  H      0.000000    0.762889   -0.470831\n"
           "  H      0.000000   -0.762889   -0.470831\n\n"
           "       The optimization did not converge but reached the maximum number of\n"
           "       optimization cycles.\n   ****ORCA TERMINATED NORMALLY****\n")
MULT_OUT = ("[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is "
            "even and number of electrons (10) is even -> impossible\n")
BASIS_OUT = ("            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
             "                                 INPUT ERROR\n"
             "            UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE\n"
             "              DEF2-SVPP \n"
             "            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")


def test_scf_fix_adds_slowconv_and_raises_maxiter():
    r = make_fix(SCF_INP, SCF_OUT)
    assert r.text.splitlines()[0] == "! B3LYP def2-SVP SlowConv"
    assert "%scf MaxIter 500 end" in r.text
    assert "%maxcore 500" in r.text and f"* xyz 0 1\n{EQ}\n*" in r.text
    assert len(r.changes) == 2 and "-%scf MaxIter 2 end" in r.diff


def test_scf_fix_escalates_to_veryslowconv():
    inp = SCF_INP.replace("def2-SVP", "def2-SVP SlowConv").replace("MaxIter 2", "MaxIter 800")
    r = make_fix(inp, SCF_OUT)
    assert "VerySlowConv" in r.text and "MaxIter 800" in r.text      # larger value kept
    assert r.changes == ["! line: SlowConv -> VerySlowConv"]


def test_scf_fix_when_already_at_the_strongest_setting():
    inp = SCF_INP.replace("def2-SVP", "def2-SVP VerySlowConv").replace("MaxIter 2", "MaxIter 900")
    r = make_fix(inp, SCF_OUT)
    assert r.text is None and any("already set" in s for s in r.skipped)


def test_scf_fix_inserts_missing_block_before_coordinates():
    inp = f"! B3LYP def2-SVP\n* xyz 0 1\n{EQ}\n*\n"
    r = make_fix(inp, SCF_OUT)
    assert "%scf MaxIter 500 end\n* xyz 0 1" in r.text


def test_scf_fix_multiline_block():
    inp = f"! B3LYP\n%scf\n  MaxIter 100\n  Shift Shift 0.3 ErrOff 0.1\nend\n* xyz 0 1\n{EQ}\n*\n"
    r = make_fix(inp, SCF_OUT)
    assert "MaxIter 500" in r.text and "Shift Shift 0.3 ErrOff 0.1" in r.text


def test_opt_fix_restarts_from_last_geometry_with_more_cycles():
    r = make_fix(OPT_INP, OPT_OUT)
    assert "%geom MaxIter 200 end" in r.text
    assert "O      0.00000000     0.00000000     0.12066200" in r.text
    assert "H      0.00000000     0.76288900    -0.47083100" in r.text
    assert "O 0.0 0.0 0.117" not in r.text
    assert r.text.rstrip().endswith("*")


def test_opt_fix_doubles_a_large_limit():
    r = make_fix(OPT_INP.replace("MaxIter 1 ", "MaxIter 150 "), OPT_OUT)
    assert "MaxIter 300" in r.text


def test_opt_fix_uses_the_given_xyz_file():
    xyz = "3\nlast\nO 0.0 0.0 0.2\nH 0.0 0.8 -0.5\nH 0.0 -0.8 -0.5\n"
    r = make_fix(OPT_INP, OPT_OUT, geometry_text=xyz)
    assert "0.20000000" in r.text and "the given xyz file" in r.changes[0]


def test_opt_fix_refuses_mismatched_atoms_or_elements():
    two_atoms = OPT_OUT.replace("  H      0.000000   -0.762889   -0.470831\n", "")
    r = make_fix(OPT_INP, two_atoms)
    assert r.text is None and any("atoms" in s for s in r.skipped)
    swapped = OPT_OUT.replace("  O  ", "  N  ")
    r = make_fix(OPT_INP, swapped)
    assert r.text is None and any("element order" in s for s in r.skipped)


def test_opt_fix_needs_an_inline_xyz_block():
    r = make_fix("! B3LYP Opt\n* xyzfile 0 1 mol.xyz\n", OPT_OUT)
    assert r.text is None and any("inline" in s for s in r.skipped)


def test_multiplicity_is_not_changed_without_an_explicit_request():
    r = make_fix(MULT_INP, MULT_OUT)
    assert r.text is None
    assert any("Valid values for 10 electrons: 1, 3, 5, 7, 9" in s for s in r.skipped)


def test_multiplicity_changed_when_requested_and_valid():
    r = make_fix(MULT_INP, MULT_OUT, multiplicity=1)
    assert "* xyz 0 1\n" in r.text and r.changes == ["multiplicity: 2 -> 1 (charge 0 unchanged)"]
    assert make_fix(MULT_INP, MULT_OUT, multiplicity=3).text is not None


def test_invalid_requested_multiplicity_is_refused():
    r = make_fix(MULT_INP, MULT_OUT, multiplicity=2)
    assert r.text is None and any("impossible" in s for s in r.skipped)


def test_multiplicity_option_without_such_an_error_is_ignored():
    r = make_fix(SCF_INP, SCF_OUT, multiplicity=3)
    assert any("ignored" in s for s in r.skipped) and "0 1" in r.text


def test_unknown_keyword_gets_suggestions_but_no_change():
    r = make_fix(BASIS_INP, BASIS_OUT)
    assert r.text is None
    assert any("DEF2-SVPP: did you mean def2-SVP" in s for s in r.skipped)


def test_unrecognisable_keyword_has_no_suggestion():
    r = make_fix(BASIS_INP, BASIS_OUT.replace("DEF2-SVPP", "QWERTYUIOP"))
    assert any("no similar common keyword" in s for s in r.skipped)


def test_imaginary_frequencies_are_reported_as_not_automated():
    out = ("VIBRATIONAL FREQUENCIES\n-----\n\n     5:   -1383.37 cm**-1  ***imaginary mode***\n"
           "     7:    3652.51 cm**-1\n\n   ****ORCA TERMINATED NORMALLY****\n")
    r = make_fix("! B3LYP Freq\n* xyz 0 1\nH 0 0 0\n*\n", out)
    assert r.text is None and any("not automated" in s for s in r.skipped)


def test_clean_output_has_nothing_to_fix():
    r = make_fix(SCF_INP, "   ****ORCA TERMINATED NORMALLY****\n")
    assert r.text is None and r.changes == [] and r.skipped == []


def test_inputs_are_never_modified_in_place():
    before = SCF_INP
    make_fix(SCF_INP, SCF_OUT)
    assert SCF_INP == before


def test_next_fix_path_never_overwrites(tmp_path):
    inp = tmp_path / "job.inp"
    inp.write_text("x")
    assert next_fix_path(inp).name == "job_fix1.inp"
    (tmp_path / "job_fix1.inp").write_text("x")
    (tmp_path / "job_fix2.inp").write_text("x")
    assert next_fix_path(inp).name == "job_fix3.inp"


# Checks against the real outputs of the failing runs (kept in tests/data).
def real(name):
    path = DATA / f"fail_{name}.out"
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    return path.read_text(errors="replace")


def test_real_scf_output_gets_the_fix():
    r = make_fix(SCF_INP, real("scf_fail"))
    assert "SlowConv" in r.text and "%scf MaxIter 500 end" in r.text


def test_real_opt_output_gets_a_restart_geometry():
    r = make_fix(OPT_INP, real("opt_maxiter"))
    assert "%geom MaxIter 200 end" in r.text and "coordinates replaced" in r.changes[0]


def test_real_multiplicity_output():
    assert make_fix(MULT_INP, real("bad_mult"), multiplicity=1).text is not None


def test_real_basis_output_suggests_def2_svp():
    r = make_fix(BASIS_INP, real("bad_basis"))
    assert any("did you mean def2-SVP" in s for s in r.skipped)


def test_real_output_echoes_the_input_with_pipe_prefix():
    # diagnose relies on this format to recognise transition-state jobs
    assert "|  1> ! B3LYP def2-SVP" in real("scf_fail")
