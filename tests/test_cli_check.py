import json

from orca_assistant.cli import main

GOOD = "! B3LYP def2-SVP\n%pal nprocs 1 end\n%maxcore 300\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n"
BAD = GOOD.replace("0 1", "0 2")


def test_check_exit_codes(tmp_path, capsys):
    ok, bad = tmp_path / "ok.inp", tmp_path / "bad.inp"
    ok.write_text(GOOD)
    bad.write_text(BAD)
    assert main(["check", str(ok)]) == 0
    assert main(["check", str(bad)]) == 1
    assert "[ERROR] multiplicity 2 is impossible with 2 electrons" in capsys.readouterr().out


def test_check_json(tmp_path, capsys):
    bad = tmp_path / "bad.inp"
    bad.write_text(BAD)
    assert main(["check", str(bad), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert any(d["code"] == "bad_multiplicity" and d["level"] == "error" for d in data)


def test_check_missing_file(tmp_path, capsys):
    assert main(["check", str(tmp_path / "no.inp")]) == 3
    assert "file not found" in capsys.readouterr().err
