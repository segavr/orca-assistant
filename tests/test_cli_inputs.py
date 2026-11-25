import json
import os
import sys

import pytest

from orca_assistant.cli import main

GOOD_INP = (
    "! B3LYP def2-SVP Opt\n"
    "%pal nprocs 2 end\n"
    "%maxcore 1000\n"
    "* xyz 0 1\n"
    "O 0 0 0.117\n"
    "H 0 0.757 -0.469\n"
    "H 0 -0.757 -0.469\n"
    "*\n"
)

SPEC = {
    "molecule": {"atoms": [
        {"element": "O", "x": 0, "y": 0, "z": 0.117},
        {"element": "H", "x": 0, "y": 0.757, "z": -0.469},
        {"element": "H", "x": 0, "y": -0.757, "z": -0.469}]},
    "resources": {"nprocs": 4, "maxcore_mb": 2000},
}


@pytest.mark.parametrize("command", ["parse", "xyz", "table"])
def test_missing_input_is_one_clean_line(command, tmp_path, capsys):
    missing = tmp_path / "nope.out"
    assert main([command, str(missing)]) == 2
    err = capsys.readouterr().err
    assert "not found" in err and "Traceback" not in err


def test_render_missing_spec(tmp_path, capsys):
    assert main(["render", str(tmp_path / "nope.json")]) == 2
    assert "spec file not found" in capsys.readouterr().err


def test_render_malformed_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert main(["render", str(path)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_render_invalid_spec_names_the_field(tmp_path, capsys):
    spec = json.loads(json.dumps(SPEC))
    spec["molecule"]["multiplicity"] = 2
    path = tmp_path / "s.json"
    path.write_text(json.dumps(spec))
    assert main(["render", str(path)]) == 2
    err = capsys.readouterr().err
    assert "not a valid CalcSpec" in err and "multiplicity" in err
    assert "Traceback" not in err


def test_check_clean_file(tmp_path, capsys):
    path = tmp_path / "job.inp"
    path.write_text(GOOD_INP)
    assert main(["check", str(path), "--no-hardware"]) == 0
    out = capsys.readouterr().out
    assert "0 error(s), 0 warning(s)" in out
    assert "nprocs=2, maxcore=1000 MB per core" in out


def test_check_broken_file_exits_two(tmp_path, capsys):
    path = tmp_path / "job.inp"
    path.write_text(GOOD_INP.replace("*\n", ""))
    assert main(["check", str(path), "--no-hardware"]) == 2
    assert "coordinate block is never closed" in capsys.readouterr().out


def test_check_json(tmp_path, capsys):
    path = tmp_path / "job.inp"
    path.write_text(GOOD_INP)
    assert main(["check", str(path), "--json", "--no-hardware"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_atoms"] == 3 and payload["nprocs"] == 2
    assert payload["issues"] == []


def test_batch_from_inp(tmp_path):
    inp = tmp_path / "water.inp"
    inp.write_text(GOOD_INP)
    script = tmp_path / "run.sh"
    assert main(["batch", str(inp), "-o", str(script), "--partition", "cpu",
                 "--module", "orca/6.1.1"]) == 0
    text = script.read_text()
    assert "#SBATCH --ntasks-per-node=2" in text
    assert "#SBATCH --mem-per-cpu=1000M" in text
    assert "orca water.inp > water.out" in text


def test_batch_from_spec_uses_the_rendered_name(tmp_path, capsys):
    spec = tmp_path / "benzene.json"
    spec.write_text(json.dumps(SPEC))
    script = tmp_path / "run.pbs"
    assert main(["batch", str(spec), "--scheduler", "pbs", "-o", str(script)]) == 0
    text = script.read_text()
    assert "#PBS -l nodes=1:ppn=4" in text
    assert "#PBS -l mem=8000mb" in text
    assert "orca benzene.inp > benzene.out" in text


def test_batch_refuses_a_broken_input(tmp_path, capsys):
    inp = tmp_path / "job.inp"
    inp.write_text("! B3LYP\n")
    script = tmp_path / "run.sh"
    assert main(["batch", str(inp), "-o", str(script)]) == 2
    assert not script.exists()
    assert "would not run" in capsys.readouterr().err


def test_batch_bad_time_is_a_cli_error(tmp_path, capsys):
    inp = tmp_path / "job.inp"
    inp.write_text(GOOD_INP)
    assert main(["batch", str(inp), "--time", "24h"]) == 2
    assert "time limit" in capsys.readouterr().err


def test_batch_prints_warnings_for_missing_maxcore(tmp_path, capsys):
    inp = tmp_path / "job.inp"
    inp.write_text(GOOD_INP.replace("%maxcore 1000\n", ""))
    assert main(["batch", str(inp)]) == 0
    captured = capsys.readouterr()
    assert "%maxcore" in captured.err
    assert "--mem-per-cpu" not in captured.out


def test_parse_reports_no_optimisation_instead_of_none(tmp_path, capsys):
    out = tmp_path / "sp.out"
    out.write_text("FINAL SINGLE POINT ENERGY      -76.400000000000\n"
                   "ORCA TERMINATED NORMALLY\n")
    assert main(["parse", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "optimisation        : n/a (no optimisation)" in printed


def test_closed_pipe_is_not_a_traceback(tmp_path, capsys):
    """`render | head` must not end in BrokenPipeError."""
    atoms = [{"element": "H", "x": i * 0.5, "y": 0.0, "z": 0.0} for i in range(1000)]
    spec = tmp_path / "big.json"
    spec.write_text(json.dumps({"molecule": {"atoms": atoms, "multiplicity": 1}}))
    read_end, write_end = os.pipe()
    os.close(read_end)  # nobody reads this pipe
    broken = os.fdopen(write_end, "w")
    original = sys.stdout
    sys.stdout = broken
    try:
        code = main(["render", str(spec)])
    finally:
        sys.stdout = original
        try:  # the pending error is re-raised on close; nothing to clean up here
            broken.close()
        except BrokenPipeError:
            pass
    assert code == 0
    captured = capsys.readouterr()
    assert "BrokenPipeError" not in captured.err
