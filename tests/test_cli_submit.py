from orca_assistant.cli import main

INP = "! B3LYP def2-SVP\n%pal nprocs 2 end\n%maxcore 1000\n* xyz 0 1\nH 0 0 0\n*\n"


def test_submit_writes_script_next_to_input(tmp_path, capsys):
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["submit", str(inp), "--scheduler", "slurm", "--partition", "short"]) == 0
    text = (tmp_path / "j.slurm").read_text()
    assert text.startswith("#!/bin/bash") and "#SBATCH --ntasks=2" in text
    assert "sbatch" in capsys.readouterr().err


def test_submit_pbs_custom_output(tmp_path):
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    dest = tmp_path / "x.sh"
    assert main(["submit", str(inp), "--scheduler", "pbs", "-o", str(dest)]) == 0
    assert "#PBS -l select=1:ncpus=2" in dest.read_text()


def test_submit_invalid_time_is_an_error(tmp_path, capsys):
    inp = tmp_path / "j.inp"
    inp.write_text(INP)
    assert main(["submit", str(inp), "--scheduler", "slurm", "--time", "soon"]) == 1
    assert "error" in capsys.readouterr().err
    assert not (tmp_path / "j.slurm").exists()


def test_submit_missing_input(tmp_path, capsys):
    assert main(["submit", str(tmp_path / "no.inp"), "--scheduler", "slurm"]) == 3
    assert "input file not found" in capsys.readouterr().err


def test_submit_flag_without_scheduler_command(tmp_path, monkeypatch, capsys):
    inp = tmp_path / "j.inp"
    inp.write_text(INP)

    def no_sbatch(script, scheduler):
        raise RuntimeError("sbatch not found in PATH")

    monkeypatch.setattr("orca_assistant.cli.submit_script", no_sbatch)
    assert main(["submit", str(inp), "--scheduler", "slurm", "--submit"]) == 1
    assert "sbatch not found" in capsys.readouterr().err
