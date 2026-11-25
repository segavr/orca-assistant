from orca_assistant.cli import main


def test_queue_without_orca_in_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("orca_assistant.cli.shutil.which", lambda _: None)
    assert main(["queue", str(tmp_path / "x.inp")]) == 2
    assert "ORCA not found" in capsys.readouterr().err
