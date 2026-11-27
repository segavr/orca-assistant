from orca_assistant.cli import main

INP = ("! B3LYP def2-SVP\n%scf MaxIter 2 end\n* xyz 0 1\n"
       "O 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n*\n")
OK_OUT = "FINAL SINGLE POINT ENERGY   -76.3\n   ****ORCA TERMINATED NORMALLY****\n"
FAIL_OUT = "SCF NOT CONVERGED AFTER   1 CYCLES\nORCA finished by error termination in LEANSCF\n"


def fake_runner_factory(_orca):
    def run(inp):
        inp.with_suffix(".out").write_text(OK_OUT if "SlowConv" in inp.read_text() else FAIL_OUT)
        return 0
    return run


def test_auto_fixes_and_reruns(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.shutil.which", lambda _: "/fake/orca")
    monkeypatch.setattr("orca_assistant.cli.default_runner", fake_runner_factory)
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["auto", str(inp)]) == 0
    out = capsys.readouterr().out
    assert "attempt 2: j_fix1.inp" in out and "finished without problems" in out
    assert (tmp_path / "j_auto1.md").exists() and inp.read_text() == INP


def test_auto_exit_code_when_the_limit_is_hit(tmp_path, monkeypatch):
    monkeypatch.setattr("orca_assistant.cli.shutil.which", lambda _: "/fake/orca")
    monkeypatch.setattr("orca_assistant.cli.default_runner",
                        lambda _o: (lambda inp: inp.with_suffix(".out").write_text(FAIL_OUT) and 0))
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["auto", str(inp), "--max-attempts", "1"]) == 1


def test_auto_without_orca(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.shutil.which", lambda _: None)
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["auto", str(inp)]) == 2
    assert "ORCA not found" in capsys.readouterr().err


def test_auto_missing_input_and_bad_limit(tmp_path, monkeypatch, capsys):
    assert main(["auto", str(tmp_path / "no.inp")]) == 3
    monkeypatch.setattr("orca_assistant.cli.shutil.which", lambda _: "/fake/orca")
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["auto", str(inp), "--max-attempts", "99"]) == 1
    assert "between 1 and 10" in capsys.readouterr().err
