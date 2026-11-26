from pathlib import Path

import pytest

from orca_assistant.diagnose import diagnose, format_findings

DATA = Path(__file__).parent / "data"

# Fragments below are verbatim from real ORCA 6.1.1 outputs.
BAD_BASIS = """
  ***********************************


            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                 INPUT ERROR
            UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE
              DEF2-SVPP 
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
[file orca_main/main_input_keywordline.cpp, line 13872]: 

[file orca_main/main_input_keywordline.cpp, line 13872]: 

"""

BAD_MULT = """
Atom   2H    basis set group =>   2
[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is even and number of electrons (10) is even -> impossible

[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is even and number of electrons (10) is even -> impossible

"""

SCF_FAIL = """
               *                      ERROR                        *
               *        SCF NOT CONVERGED AFTER   1 CYCLES         *
[file orca_leanscf/orca_leanscf.cpp, line 306]: Error (ORCA_LEANSCF): unfortunately, the SCF has not converged. There may be a way out but we have to stop here 

ORCA finished by error termination in LEANSCF
Calling Command: /home/simon/orca/orca_6_1_1_linux_x86-64_shared_openmpi418/orca_leanscf scf_fail.gbw 
[file orca_tools/qcmsg.cpp, line 394]: 
  .... aborting the run

"""

OPT_MAX = """
The optimization has not yet converged - more geometry cycles are needed

    ----------------------------------------------------------------------------
                                  WARNING !!!
       The optimization did not converge but reached the maximum number of
       optimization cycles.
       Please check your results very carefully.
    ----------------------------------------------------------------------------

                             ****ORCA TERMINATED NORMALLY****
"""

IMAG = """
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:   -1383.37 cm**-1  ***imaginary mode***
     6:   -1383.37 cm**-1  ***imaginary mode***
     7:    3652.51 cm**-1
     8:    4022.06 cm**-1


------------
                             ****ORCA TERMINATED NORMALLY****
"""


def codes(text):
    return [f.code for f in diagnose(text)]


def test_unknown_keyword_names_the_keyword():
    (f,) = diagnose(BAD_BASIS)
    assert f.code == "unknown_keyword" and f.severity == "error"
    assert f.detail == "DEF2-SVPP"


def test_bad_multiplicity_suggests_valid_values():
    (f,) = diagnose(BAD_MULT)
    assert f.code == "bad_multiplicity"
    assert "multiplicity 2 with 10 electrons" in f.detail
    assert "1, 3, 5, 7" in f.suggestion


def test_multiplicity_for_odd_electron_count():
    text = BAD_MULT.replace("(2) is even", "(1) is odd").replace("(10) is even", "(9) is odd")
    (f,) = diagnose(text)
    assert "2, 4, 6" in f.suggestion


def test_scf_failure_with_error_termination_is_one_error():
    (f,) = diagnose(SCF_FAIL)
    assert f.code == "scf_not_converged" and f.severity == "error"
    assert "after 1 cycles" in f.detail


def test_scf_warning_when_job_still_finished():
    text = "SCF NOT CONVERGED AFTER 125 CYCLES\n" + "ORCA TERMINATED NORMALLY\n"
    (f,) = diagnose(text)
    assert f.code == "scf_not_converged" and f.severity == "warning"


def test_optimisation_cycle_limit():
    (f,) = diagnose(OPT_MAX)
    assert f.code == "opt_not_converged" and f.severity == "warning"
    assert "MaxIter" in f.suggestion


def test_normal_between_cycles_message_is_not_a_finding():
    assert diagnose("The optimization has not yet converged - more geometry cycles "
                    "are needed\nORCA TERMINATED NORMALLY\n") == []


def test_large_degenerate_imaginary_pair():
    (f,) = diagnose(IMAG)
    assert f.code == "imaginary_frequencies"
    assert f.detail == "-1383.37, -1383.37 cm-1" and "2 imaginary" in f.title
    assert "degenerate" in f.suggestion


def test_small_imaginary_frequency_hint():
    text = IMAG.replace("-1383.37 cm**-1  ***imaginary mode***", "-23.10 cm**-1  ***imaginary mode***", 1)
    text = text.replace("-1383.37 cm**-1  ***imaginary mode***", "   5.00 cm**-1", 1)
    (f,) = diagnose(text)
    assert "1 imaginary frequency in" in f.title and "TightOpt" in f.suggestion


def test_ts_with_one_imaginary_frequency_is_fine():
    text = "|  1> ! B3LYP def2-SVP OptTS Freq\n" + IMAG.replace(
        "   6:   -1383.37 cm**-1  ***imaginary mode***", "   6:    1000.00 cm**-1")
    assert diagnose(text) == []


def test_ts_with_wrong_number_of_imaginary_frequencies():
    text = "|  1> ! B3LYP def2-SVP OptTS Freq\n" + IMAG
    (f,) = diagnose(text)
    assert f.code == "ts_imaginary_count"


def test_minimum_without_imaginary_frequencies_is_fine():
    text = IMAG.replace("-1383.37 cm**-1  ***imaginary mode***", "  900.00 cm**-1")
    assert diagnose(text) == []


def test_unknown_error_termination_falls_back_to_generic():
    text = ("[file orca_x/y.cpp, line 1]: Something unexpected happened\n"
            "ORCA finished by error termination in GTOInt\n")
    (f,) = diagnose(text)
    assert f.code == "error_termination" and "GTOInt" in f.title
    assert f.detail == "Something unexpected happened"


def test_truncated_output_is_flagged_with_hedge():
    (f,) = diagnose("some ORCA output that just stops\nSCF ITERATIONS\n")
    assert f.code == "truncated_output" and "guess" in f.suggestion


def test_empty_output():
    assert codes("") == ["empty_output"] and codes("  \n") == ["empty_output"]


def test_clean_finished_output():
    assert diagnose("blah\n   ****ORCA TERMINATED NORMALLY****\n") == []
    assert format_findings([]) == "no problems found\n"


def test_errors_are_listed_before_warnings():
    text = BAD_MULT + OPT_MAX
    assert [f.severity for f in diagnose(text)] == ["error", "warning"]


def test_format_findings_text():
    out = format_findings(diagnose(BAD_BASIS))
    assert out.startswith("[ERROR] Unrecognised or duplicated keyword")
    assert "evidence  : DEF2-SVPP" in out and "suggestion:" in out


@pytest.mark.parametrize("name, expected", [
    ("bad_basis", ["unknown_keyword"]),
    ("bad_mult", ["bad_multiplicity"]),
    ("scf_fail", ["scf_not_converged"]),
    ("opt_maxiter", ["opt_not_converged"]),
    ("imag_freq", ["imaginary_frequencies"]),
    ("low_mem", []),
])
def test_real_failing_outputs(name, expected):
    path = DATA / f"fail_{name}.out"
    if not path.exists():
        pytest.skip(f"{path.name} missing")
    assert codes(path.read_text(errors="replace")) == expected
