import pytest

from orca_assistant.autorun import (
    format_journal, format_summary, next_journal_path, run_with_retries,
)

EQ = "O 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469"
INP = f"! B3LYP def2-SVP\n%scf MaxIter 2 end\n* xyz 0 1\n{EQ}\n*\n"

OK_OUT = "FINAL SINGLE POINT ENERGY      -76.321078000000\n   ****ORCA TERMINATED NORMALLY****\n"
SCF_FAIL_OUT = ("               *        SCF NOT CONVERGED AFTER   1 CYCLES         *\n"
                "ORCA finished by error termination in LEANSCF\n")
MULT_OUT = ("[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is "
            "even and number of electrons (10) is even -> impossible\n")


def make_runner(decide):
    """decide(input_text, call_number) -> text of the fake .out"""
    calls = []

    def runner(inp):
        calls.append(inp.name)
        inp.with_suffix(".out").write_text(decide(inp.read_text(), len(calls)))
        return 0

    runner.calls = calls
    return runner


def setup(tmp_path, text=INP):
    p = tmp_path / "job.inp"
    p.write_text(text)
    return p


def quiet(_):
    pass


def test_success_on_the_first_attempt_changes_nothing(tmp_path):
    inp = setup(tmp_path)
    runner = make_runner(lambda t, n: OK_OUT)
    res = run_with_retries(inp, runner, log=quiet)
    assert res.status == "ok" and len(res.attempts) == 1
    assert res.attempts[0].energy_eh == pytest.approx(-76.321078)
    assert not (tmp_path / "job_fix1.inp").exists() and inp.read_text() == INP


def test_scf_failure_is_fixed_and_rerun(tmp_path):
    inp = setup(tmp_path)
    runner = make_runner(lambda t, n: OK_OUT if "SlowConv" in t else SCF_FAIL_OUT)
    res = run_with_retries(inp, runner, log=quiet)
    assert res.status == "ok" and runner.calls == ["job.inp", "job_fix1.inp"]
    first = res.attempts[0]
    assert first.findings[0].code == "scf_not_converged"
    assert first.next_inp == tmp_path / "job_fix1.inp" and first.changes
    assert inp.read_text() == INP                                   # original untouched
    assert "SlowConv" in (tmp_path / "job_fix1.inp").read_text()


def test_stops_when_a_human_decision_is_needed(tmp_path):
    inp = setup(tmp_path, INP.replace("0 1", "0 2"))
    runner = make_runner(lambda t, n: MULT_OUT)
    res = run_with_retries(inp, runner, log=quiet)
    assert res.status == "needs_human" and len(res.attempts) == 1
    assert any("Multiplicity" in s for s in res.attempts[0].skipped)
    assert not (tmp_path / "job_fix1.inp").exists()


def test_imaginary_frequencies_stop_the_loop(tmp_path):
    out = ("VIBRATIONAL FREQUENCIES\n-----\n\n     5:   -1383.37 cm**-1  ***imaginary mode***\n"
           "     7:    3652.51 cm**-1\n\n   ****ORCA TERMINATED NORMALLY****\n")
    res = run_with_retries(setup(tmp_path), make_runner(lambda t, n: out), log=quiet)
    assert res.status == "needs_human" and len(res.attempts) == 1


def test_attempt_limit(tmp_path):
    runner = make_runner(lambda t, n: SCF_FAIL_OUT)
    res = run_with_retries(setup(tmp_path), runner, max_attempts=2, log=quiet)
    assert res.status == "limit" and runner.calls == ["job.inp", "job_fix1.inp"]


def test_stops_when_nothing_more_can_be_fixed(tmp_path):
    runner = make_runner(lambda t, n: SCF_FAIL_OUT)
    res = run_with_retries(setup(tmp_path), runner, max_attempts=6, log=quiet)
    # SlowConv, then VerySlowConv, then the input is already at the strongest setting
    assert res.status == "needs_human" and len(res.attempts) == 3
    assert "VerySlowConv" in (tmp_path / "job_fix2.inp").read_text()
    assert any("already set" in s for s in res.attempts[-1].skipped)


def test_existing_fix_files_are_not_overwritten(tmp_path):
    (tmp_path / "job_fix1.inp").write_text("precious")
    runner = make_runner(lambda t, n: OK_OUT if "SlowConv" in t else SCF_FAIL_OUT)
    res = run_with_retries(setup(tmp_path), runner, log=quiet)
    assert (tmp_path / "job_fix1.inp").read_text() == "precious"
    assert runner.calls == ["job.inp", "job_fix2.inp"] and res.status == "ok"


def test_missing_output_is_reported_not_crashing(tmp_path):
    res = run_with_retries(setup(tmp_path), lambda inp: 1, log=quiet)
    assert res.status == "needs_human"
    assert res.attempts[0].findings[0].code == "empty_output"


def test_interrupt_keeps_the_attempts_so_far(tmp_path):
    def decide(text, n):
        if n == 2:
            raise KeyboardInterrupt
        return SCF_FAIL_OUT

    res = run_with_retries(setup(tmp_path), make_runner(decide), log=quiet)
    assert res.status == "interrupted" and len(res.attempts) == 1


@pytest.mark.parametrize("n", [0, 11, -1])
def test_attempt_limit_is_validated(tmp_path, n):
    with pytest.raises(ValueError):
        run_with_retries(setup(tmp_path), lambda i: 0, max_attempts=n, log=quiet)


def test_summary_and_journal(tmp_path):
    inp = setup(tmp_path)
    runner = make_runner(lambda t, n: OK_OUT if "SlowConv" in t else SCF_FAIL_OUT)
    res = run_with_retries(inp, runner, log=quiet)
    summary = format_summary(res)
    assert "attempt 1: job.inp" in summary and "SCF did not converge" in summary
    assert summary.rstrip().endswith("result: finished without problems (2 attempts)")
    journal = format_journal(res, inp, when="2026-11-26 12:00")
    assert journal.startswith("# Automatic run: job.inp")
    assert "| 2 | job_fix1.inp |" in journal and "-76.32107800" in journal
    assert "```diff" in journal and "+%scf MaxIter 500 end" in journal
    assert "started: 2026-11-26 12:00" in journal


def test_journal_lists_what_was_left_to_the_user(tmp_path):
    inp = setup(tmp_path, INP.replace("0 1", "0 2"))
    res = run_with_retries(inp, make_runner(lambda t, n: MULT_OUT), log=quiet)
    journal = format_journal(res, inp, when="x")
    assert "Not changed automatically:" in journal and "Valid values" in journal


def test_journal_paths_do_not_collide(tmp_path):
    inp = setup(tmp_path)
    assert next_journal_path(inp).name == "job_auto1.md"
    (tmp_path / "job_auto1.md").write_text("x")
    assert next_journal_path(inp).name == "job_auto2.md"
