from orca_assistant.cli import _orca_path, main
from orca_assistant.orcafind import OrcaSearch


def test_doctor_with_orca(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.find_orca",
                        lambda: OrcaSearch("/opt/orca/orca", "real", "12 orca_* programs next to it"))
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "/opt/orca/orca  (confirmed: 12 orca_* programs next to it)" in out
    assert "cores / RAM" in out and "Python" in out


def test_doctor_without_orca(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.find_orca", lambda: OrcaSearch(None, "none"))
    assert main(["doctor"]) == 1
    assert "NOT FOUND" in capsys.readouterr().out


def test_orca_path_explains_what_was_skipped(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.find_orca", lambda: OrcaSearch(
        None, "none", skipped=["/usr/bin/orca: a script (the GNOME screen reader)"]))
    assert _orca_path() is None
    err = capsys.readouterr().err
    assert "skipped /usr/bin/orca" in err and "orca-assistant doctor" in err


def test_orca_path_is_quiet_for_a_confirmed_orca(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.find_orca",
                        lambda: OrcaSearch("/opt/orca/orca", "real", "x", ["/usr/bin/orca: script"]))
    assert _orca_path() == "/opt/orca/orca"
    assert capsys.readouterr().err == ""


def test_orca_path_warns_about_an_unconfirmed_one(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.find_orca",
                        lambda: OrcaSearch("/x/orca", "unconfirmed", "no helpers"))
    assert _orca_path() == "/x/orca"
    assert "nothing next to it confirms" in capsys.readouterr().err
