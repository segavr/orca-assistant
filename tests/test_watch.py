import io
from pathlib import Path

import pytest

from orca_assistant.watch import format_status, parse_progress, watch

WATER = Path(__file__).parent / "data" / "water.out"
needs_water = pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")

HEAD = "Program Version 6.1.1  -  RELEASE  -\n"
SCF_RUN = (HEAD + "GEOMETRY OPTIMIZATION CYCLE   1\nSCF ITERATIONS\n"
           "ITER       Energy         Delta-E        Max-DP      RMS-DP      [F,P]     Damp\n"
           "   0   -75.900000000000   0.000000000000   0.0211   0.0067   0.1830   0.7000\n"
           "   1   -76.300000000000  -0.400000000000   0.0100   0.0030   0.0900   0.7000\n")
CYCLE1 = SCF_RUN + "SCF CONVERGED AFTER   2 CYCLES\nFINAL SINGLE POINT ENERGY      -76.400000000000\n"
CONV = ("                    |Geometry convergence|\n"
        "          Item                value                   Tolerance       Converged\n"
        "          RMS gradient        0.0065547413            0.0001000000      NO\n"
        "          MAX gradient        0.0080135197            0.0003000000      NO\n"
        "          RMS step            0.0125132965            0.0020000000      NO\n"
        "          MAX step            0.0152532948            0.0040000000      NO\n")
DONE = (CYCLE1 + CONV + "THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY\n"
        "TOTAL RUN TIME: 0 days 0 hours 1 minutes 5 seconds 500 msec\n")


def test_empty_is_starting():
    assert parse_progress("").state == "starting"


def test_scf_in_progress():
    p = parse_progress(SCF_RUN)
    assert p.state == "scf" and p.scf_iteration == 1 and p.opt_cycle == 1
    assert p.scf_delta_e == -0.4 and p.energy_eh is None and p.orca_version == "6.1.1"


def test_after_scf_is_optimisation_with_convergence_table():
    p = parse_progress(CYCLE1 + CONV)
    assert p.state == "optimisation" and p.energy_eh == -76.4
    assert [c.name for c in p.convergence] == ["RMS gradient", "MAX gradient",
                                               "RMS step", "MAX step"]
    assert p.convergence[0].value == 0.0065547413 and not p.convergence[0].converged


def test_converged_then_finished():
    assert parse_progress(CYCLE1 + CONV + "THE OPTIMIZATION HAS CONVERGED\n").state == "frequencies"
    p = parse_progress(DONE)
    assert p.state == "finished" and p.runtime_s == 65.5 and p.opt_converged


def test_error_termination_is_failure():
    p = parse_progress(SCF_RUN + "ORCA finished by error termination in SCF\n")
    assert p.state == "failed" and any("SCF" in n for n in p.notes)
    assert "FAILED" in format_status(p)


def test_scf_not_converged_is_noted():
    p = parse_progress(SCF_RUN + "SCF NOT CONVERGED AFTER 125 CYCLES\n")
    assert "SCF did not converge" in p.notes


@needs_water
def test_water_complete_file():
    p = parse_progress(WATER.read_text())
    assert p.state == "finished" and p.orca_version == "6.1.1"
    assert p.energy_eh == -76.437322517640 and p.opt_converged
    assert abs(p.runtime_s - 17.282) < 1e-6
    assert p.convergence
    flags = {c.name: c.converged for c in p.convergence}
    assert all(v for k, v in flags.items() if k != "Energy change")


@needs_water
def test_water_cut_after_first_cycle():
    text = WATER.read_text()
    p = parse_progress(text[:text.index("GEOMETRY OPTIMIZATION CYCLE   2")])
    assert p.opt_cycle == 1 and not p.finished and p.energy_eh == -76.437204896778


@needs_water
def test_water_cut_during_second_scf():
    text = WATER.read_text()
    second_scf_end = [i for i in range(len(text)) if text.startswith("SCF CONVERGED AFTER", i)][1]
    p = parse_progress(text[:second_scf_end])
    assert p.state == "scf" and p.opt_cycle == 2 and p.scf_iteration is not None


def test_watch_follows_a_growing_file(tmp_path):
    f = tmp_path / "job.out"
    stages = iter([SCF_RUN, CYCLE1 + CONV, DONE])

    def fake_sleep(_):
        f.write_text(next(stages))

    ticks = iter(range(0, 1000, 10))
    out = io.StringIO()
    code = watch(f, interval=0, out=out, sleep=fake_sleep, clock=lambda: next(ticks), tty=False)
    text = out.getvalue()
    assert code == 0
    assert "waiting for the output file" in text and "SCF iterations" in text
    assert "finished normally" in text and "run time     : 1m 05s" in text


def test_watch_reports_failure(tmp_path):
    f = tmp_path / "job.out"
    f.write_text(SCF_RUN + "ORCA finished by error termination in SCF\n")
    assert watch(f, out=io.StringIO(), tty=False) == 1


def test_watch_once_missing_file(tmp_path):
    assert watch(tmp_path / "none.out", once=True, out=io.StringIO(), tty=False) == 2


def test_watch_gives_up(tmp_path):
    f = tmp_path / "job.out"
    f.write_text(SCF_RUN)
    assert watch(f, out=io.StringIO(), sleep=lambda _: None, max_polls=3, tty=False) == 3


def test_time_per_cycle_is_shown(tmp_path):
    f = tmp_path / "job.out"
    f.write_text(CYCLE1)
    texts = iter([CYCLE1.replace("CYCLE   1", "CYCLE   2"),
                  CYCLE1.replace("CYCLE   1", "CYCLE   3")])
    ticks = iter(range(0, 1000, 12))

    def fake_sleep(_):
        f.write_text(next(texts))

    out = io.StringIO()
    watch(f, out=out, sleep=fake_sleep, clock=lambda: next(ticks), tty=False, max_polls=3)
    assert "time / cycle" in out.getvalue()


REAL_SCF = (HEAD + "GEOMETRY OPTIMIZATION CYCLE   1\n"
            "Iteration    Energy (Eh)           Delta-E    RMSDP     MaxDP     DIISErr   Damp  Time(sec)\n"
            "    1    -76.3385851164243832     0.00e+00  1.12e-02  6.59e-02  3.43e-01  0.700   0.1\n"
            "    2    -76.3878708748528794    -4.93e-02  6.46e-03  3.92e-02  1.36e-01  0.700   0.1\n")


def test_orca6_scf_table_with_scientific_delta_e():
    p = parse_progress(REAL_SCF)
    assert p.state == "scf" and p.scf_iteration == 2 and p.scf_delta_e == -4.93e-02


def test_orbital_energy_rows_are_not_scf_iterations():
    rows = ("ORBITAL ENERGIES\n  NO   OCC          E(Eh)            E(eV)\n"
            "   0   2.0000     -19.121200      -520.3143\n")
    assert parse_progress(HEAD + rows).scf_iteration is None


def test_orca_startup_error_is_failure_not_starting():
    text = HEAD + "[file orca_main/run.cpp, line 393]: Cannot open input file: benzene.inp\n"
    p = parse_progress(text)
    assert p.state == "failed"
    assert any("Cannot open input file" in n for n in p.notes)


def test_final_scf_after_converged_optimisation_is_labelled():
    text = (HEAD + "GEOMETRY OPTIMIZATION CYCLE   1\nSCF CONVERGED AFTER   2 CYCLES\n"
            "FINAL SINGLE POINT ENERGY      -1.500000000000\nTHE OPTIMIZATION HAS CONVERGED\n"
            "    1    -76.3385851164243832     0.00e+00  1.12e-02  6.59e-02  3.43e-01  0.700   0.1\n")
    p = parse_progress(text)
    assert p.state == "scf" and p.opt_converged
    assert "final SCF after the optimisation" in format_status(p)


def test_input_error_with_empty_message_is_failure_with_reason():
    text = (HEAD + "            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "                                 INPUT ERROR\n"
            "            UNRECOGNIZED OR DUPLICATED KEYWORD(S) IN SIMPLE INPUT LINE\n"
            "              DEF2-SVPP \n"
            "            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "[file orca_main/main_input_keywordline.cpp, line 13872]: \n")
    p = parse_progress(text)
    assert p.state == "failed"
    assert any("Unrecognised or duplicated keyword" in n for n in p.notes)
