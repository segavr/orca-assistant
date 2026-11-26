import json

from orca_assistant.cli import main

BAD_MULT = ("[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is "
            "even and number of electrons (10) is even -> impossible\n")


def test_diagnose_reports_and_returns_one(tmp_path, capsys):
    f = tmp_path / "x.out"
    f.write_text(BAD_MULT)
    assert main(["diagnose", str(f)]) == 1
    out = capsys.readouterr().out
    assert "[ERROR] Charge and multiplicity do not match" in out and "1, 3, 5, 7" in out


def test_diagnose_json(tmp_path, capsys):
    f = tmp_path / "x.out"
    f.write_text(BAD_MULT)
    assert main(["diagnose", str(f), "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert data[0]["code"] == "bad_multiplicity" and data[0]["severity"] == "error"


def test_diagnose_clean_output_returns_zero(tmp_path, capsys):
    f = tmp_path / "ok.out"
    f.write_text("****ORCA TERMINATED NORMALLY****\n")
    assert main(["diagnose", str(f)]) == 0
    assert "no problems found" in capsys.readouterr().out


def test_diagnose_missing_file(tmp_path, capsys):
    assert main(["diagnose", str(tmp_path / "no.out")]) == 3
    assert "file not found" in capsys.readouterr().err
