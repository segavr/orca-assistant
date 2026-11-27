import pytest

from orca_assistant.inplint import Z, format_issues, lint_input
from orca_assistant.resources import Hardware

EQ = "O 0.000000 0.000000 0.117000\nH 0.000000 0.757000 -0.469000\nH 0.000000 -0.757000 -0.469000"
BENZENE = "\n".join(
    ["C 1.45 0.0 0.0", "C 0.725 1.2557 0.0", "C -0.725 1.2557 0.0", "C -1.45 0.0 0.0",
     "C -0.725 -1.2557 0.0", "C 0.725 -1.2557 0.0", "H 2.55 0.0 0.0", "H 1.275 2.2084 0.0",
     "H -1.275 2.2084 0.0", "H -2.55 0.0 0.0", "H -1.275 -2.2084 0.0", "H 1.275 -2.2084 0.0"])
GOOD = ("! B3LYP def2-SVP D3BJ def2/J RIJCOSX TightSCF Opt Freq\n%pal nprocs 2 end\n"
        f"%maxcore 1058\n* xyz 0 1\n{BENZENE}\n*\n")
SMALL = Hardware(cores=4, ram_mb=6835, available_mb=3100)

# The exact inputs behind the real failing outputs in tests/data/fail_*.out
SCF_INP = f"! B3LYP def2-SVP\n%maxcore 500\n%scf MaxIter 2 end\n* xyz 0 1\n{EQ}\n*\n"
OPT_INP = ("! B3LYP def2-SVP Opt\n%maxcore 500\n%geom MaxIter 1 end\n* xyz 0 1\n"
           "O 0.0 0.0 0.117\nH 0.0 1.1 -0.6\nH 0.0 -1.1 -0.6\n*\n")
MULT_INP = f"! B3LYP def2-SVP\n%maxcore 500\n* xyz 0 2\n{EQ}\n*\n"
BASIS_INP = f"! B3LYP def2-SVPP\n%maxcore 500\n* xyz 0 1\n{EQ}\n*\n"
MEM_INP = f"! B3LYP def2-TZVP\n%maxcore 10\n* xyz 0 1\n{EQ}\n*\n"


def codes(text, **kw):
    """Codes of all findings except the routine 'no %maxcore' note of tiny test inputs."""
    return [i.code for i in lint_input(text, **kw) if i.code != "no_maxcore"]


def test_element_table_is_correct():
    assert len(Z) == 118
    assert (Z["H"], Z["C"], Z["Fe"], Z["Kr"], Z["I"], Z["Xe"], Z["Au"], Z["Og"]) == \
        (1, 6, 26, 36, 53, 54, 79, 118)


def test_a_good_input_has_no_findings():
    assert lint_input(GOOD, hardware=SMALL) == []
    assert format_issues([]) == "no problems found\n"


def test_impossible_multiplicity_matches_the_real_orca_error():
    (i,) = lint_input(MULT_INP)[:1]
    assert i.code == "bad_multiplicity" and i.level == "error"
    assert "10 electrons" in i.message and "1, 3, 5, 7, 9" in i.message


def test_charge_changes_the_electron_count():
    assert "bad_multiplicity" not in codes(MULT_INP.replace("0 2", "1 2"))   # 9 electrons
    assert "bad_multiplicity" in codes(MULT_INP.replace("0 2", "-1 1"))      # 11 electrons


def test_multiplicity_larger_than_electrons_allows():
    assert "bad_multiplicity" in codes(MULT_INP.replace("0 2", "0 13"))


def test_special_atoms_skip_the_electron_check():
    inp = "! HF def2-SVP\n* xyz 0 2\nH 0 0 0\nH: 0 0 1\n*\n"
    assert codes(inp) == ["electron_count_skipped"]


def test_unknown_element_skips_the_check_with_a_note():
    inp = "! HF def2-SVP\n* xyz 0 1\nXx 0 0 0\n*\n"
    out = lint_input(inp)
    assert out[0].code == "electron_count_skipped" and "unknown element" in out[0].message


def test_unclosed_coordinates():
    assert "unclosed_coordinates" in codes("! HF def2-SVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n")


def test_missing_coordinates_and_keyword_line():
    assert "no_coordinates" in codes("! HF def2-SVP\n")
    assert "no_keyword_line" in codes("* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")


def test_missing_charge_and_multiplicity():
    assert "bad_charge_mult" in codes("! HF\n* xyz\nH 0 0 0\n*\n")
    assert "bad_charge_mult" in codes("! HF\n* xyz 0 0\nH 0 0 0\n*\n")


def test_xyzfile_must_exist(tmp_path):
    inp = "! HF def2-SVP\n* xyzfile 0 1 mol.xyz\n"
    assert "missing_xyzfile" in codes(inp, base_dir=tmp_path)
    (tmp_path / "mol.xyz").write_text("x")
    assert codes(inp, base_dir=tmp_path) == []
    assert codes(inp) == []                       # without a folder the file is not checked


def test_duplicate_keywords_across_lines():
    out = [i for i in lint_input("! B3LYP def2-SVP\n! opt B3LYP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
           if i.code != "no_maxcore"]
    assert [i.code for i in out] == ["duplicate_keyword"] and out[0].line == 2


def test_misspelt_basis_is_flagged_softly():
    out = lint_input(BASIS_INP)
    assert [i.code for i in out] == ["possible_typo"] and out[0].level == "info"
    assert "def2-SVP" in out[0].message


def test_common_valid_keywords_are_not_flagged():
    inp = ("! wB97X-D3BJ def2-TZVP VeryTightSCF TightOpt CPCM(water) def2/J SlowConv NormalSCF\n"
           "* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
    assert codes(inp) == []


def test_composite_method_with_basis_and_double_dispersion():
    assert "composite_with_basis" in codes("! r2SCAN-3c def2-TZVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
    assert "double_dispersion" in codes("! wB97X-D3 D3BJ def2-SVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
    assert "double_dispersion" not in codes("! B3LYP D3BJ def2-SVP\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")


def test_ts_without_hessian_is_an_info():
    inp = "! B3LYP def2-SVP OptTS\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n"
    assert "ts_without_hessian" in codes(inp)
    assert "ts_without_hessian" not in codes(inp.replace("OptTS", "OptTS Freq"))
    assert "ts_without_hessian" not in codes(inp.replace("\n*", "\n%geom Calc_Hess true end\n*", 1))


def test_low_iteration_limits_from_the_real_failing_runs():
    assert "low_scf_maxiter" in codes(SCF_INP)
    assert "low_geom_maxiter" in codes(OPT_INP)
    assert "low_scf_maxiter" not in codes(SCF_INP.replace("MaxIter 2", "MaxIter 200"))


def test_tiny_maxcore_from_the_real_run():
    assert "tiny_maxcore" in codes(MEM_INP)
    assert [i.code for i in lint_input("! HF\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")] == ["no_maxcore"]


def test_unclosed_block():
    inp = "! B3LYP def2-SVP\n%scf MaxIter 100\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n"
    assert "unclosed_block" in codes(inp)
    assert "unclosed_block" not in codes(inp.replace("100", "100 end"))
    assert "unclosed_block" not in codes(GOOD)            # %maxcore needs no 'end'
    assert "unclosed_block" not in codes(                  # nested 'end end' is fine
        "! Opt\n%geom Scan B 0 1 = 1.0, 2.0, 5 end end\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")


def test_comments_are_ignored():
    inp = ("# ! B3LYP B3LYP duplicated in a comment\n! B3LYP def2-SVP # trailing\n"
           "%maxcore 1000  # per core\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n")
    assert "duplicate_keyword" not in codes(inp) and "no_keyword_line" not in codes(inp)


def test_hardware_checks():
    big = GOOD.replace("nprocs 2", "nprocs 4").replace("1058", "2000")   # 8000 MB > 6835 MB
    out = lint_input(big, hardware=SMALL)
    assert out[0].code == "memory_exceeds_ram" and "Suggested for this machine" in out[0].message
    assert "too_many_procs" in codes(GOOD.replace("nprocs 2", "nprocs 8"), hardware=SMALL)
    assert "single_core" in codes(GOOD.replace("%pal nprocs 2 end\n", ""), hardware=SMALL)
    near = GOOD.replace("nprocs 2", "nprocs 4").replace("1058", "1400")   # 5600 MB > 75%
    assert "memory_near_ram" in codes(near, hardware=SMALL)
    busy = Hardware(cores=4, ram_mb=16000, available_mb=1000)
    assert "memory_exceeds_available" in codes(GOOD, hardware=busy)
    assert not any(c.startswith("memory") for c in codes(GOOD))           # no hardware: no check


def test_errors_come_first():
    out = lint_input(MULT_INP.replace("def2-SVP", "def2-SVPP"))
    assert [i.level for i in out] == ["error", "info"]


def test_format_issues_shows_levels_and_lines():
    text = format_issues(lint_input("! B3LYP def2-SVP\n! opt B3LYP\n%maxcore 1000\n* xyz 0 1\nH 0 0 0\nH 0 0 0.7\n*\n"))
    assert text.startswith("[WARNING] keyword 'B3LYP' appears twice") and "(line 2)" in text
