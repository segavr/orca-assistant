import shutil
import subprocess
import sys

import pytest

from orca_assistant.batch import BatchSpec, build_script, submit

needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None or sys.platform.startswith("win"), reason="bash required")


def make_inp(tmp_path, nprocs=4, maxcore=2000, name="job.inp", pal=True):
    p = tmp_path / name
    pal_block = f"%pal nprocs {nprocs} end\n" if pal else ""
    p.write_text(f"! B3LYP def2-SVP\n{pal_block}%maxcore {maxcore}\n* xyz 0 1\nH 0 0 0\n*\n")
    return p


def spec(tmp_path, scheduler="slurm", **kw):
    inp = kw.pop("inp", None) or make_inp(tmp_path)
    return BatchSpec(scheduler=scheduler, inp=inp, **kw)


def test_slurm_header(tmp_path):
    text, notes = build_script(spec(tmp_path, partition="short", account="proj1",
                                    time="2-12:00:00"))
    assert "#SBATCH --ntasks=4" in text and "#SBATCH --nodes=1" in text
    assert "#SBATCH --mem=10400M" in text            # 4 x 2000 x 1.3
    assert "#SBATCH --time=2-12:00:00" in text
    assert "#SBATCH --partition=short" in text and "#SBATCH --account=proj1" in text
    assert "#SBATCH --job-name=job" in text and notes == []


def test_pbs_header_and_time_conversion(tmp_path):
    text, _ = build_script(spec(tmp_path, scheduler="pbs", partition="batch", time="2-12:00:00"))
    assert "#PBS -l select=1:ncpus=4:mpiprocs=4:mem=10400mb" in text
    assert "#PBS -l walltime=60:00:00" in text       # 2 days 12 h
    assert "#PBS -q batch" in text and "#SBATCH" not in text
    assert "PBS_O_WORKDIR" in text


def test_pbs_job_name_rules(tmp_path):
    inp = make_inp(tmp_path, name="1_very_long_molecule_name_here.inp")
    text, _ = build_script(BatchSpec("pbs", inp))
    name = [l for l in text.splitlines() if l.startswith("#PBS -N")][0].split()[-1]
    assert name[0].isalpha() and len(name) <= 15


def test_explicit_memory_and_warning(tmp_path):
    text, notes = build_script(spec(tmp_path, mem_mb=4000))
    assert "#SBATCH --mem=4000M" in text
    assert any("below nprocs x maxcore" in n for n in notes)


def test_no_pal_block_is_noted(tmp_path):
    text, notes = build_script(spec(tmp_path, inp=make_inp(tmp_path, pal=False)))
    assert "#SBATCH --ntasks=1" in text and any("no %pal" in n for n in notes)


def test_modules_and_orca_path(tmp_path):
    text, _ = build_script(spec(tmp_path, modules=["gcc/12.2", "openmpi/4.1.8"],
                                orca="/opt/orca_6_1_1/orca"))
    assert "module load gcc/12.2" in text and "module load openmpi/4.1.8" in text
    assert "ORCA=/opt/orca_6_1_1/orca" in text
    assert text.index("module load") < text.index("ORCA=")


@pytest.mark.parametrize("kw", [
    {"time": "soon"}, {"time": "10:70:00"}, {"mem_mb": 0},
    {"partition": "a; rm -rf ~"}, {"account": "x y"}, {"modules": ["m; ls"]},
    {"orca": "orca"}, {"orca": "/opt/o rca/orca"}, {"scratch": "/s/$(whoami)"},
    {"scratch": "/s", "copy_files": ["a/b"]},
])
def test_invalid_values_are_rejected(tmp_path, kw):
    with pytest.raises(ValueError):
        build_script(spec(tmp_path, **kw))


def test_unknown_scheduler(tmp_path):
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_script(BatchSpec("lsf", make_inp(tmp_path)))


def test_copy_files_without_scratch_is_noted(tmp_path):
    _, notes = build_script(spec(tmp_path, copy_files=["extra.xyz"]))
    assert any("--copy-files" in n for n in notes)


@needs_bash
@pytest.mark.parametrize("scheduler", ["slurm", "pbs"])
@pytest.mark.parametrize("extra", [{}, {"scratch": "/tmp/x/$USER", "copy_files": ["g.xyz"]}])
def test_generated_script_has_valid_bash_syntax(tmp_path, scheduler, extra):
    text, _ = build_script(spec(tmp_path, scheduler=scheduler, modules=["m/1"], **extra))
    script = tmp_path / "s.sh"
    script.write_text(text)
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def fake_orca(tmp_path, body):
    d = tmp_path / "bin"
    d.mkdir()
    exe = d / "orca"
    exe.write_text("#!/bin/bash\n" + body)
    exe.chmod(0o755)
    return d


def run_script(tmp_path, text, bin_dir, env_extra, system_path=True):
    script = tmp_path / "s.sh"
    script.write_text(text)
    path = f"{bin_dir}:/usr/bin:/bin" if system_path else str(bin_dir)
    env = {"PATH": path, "HOME": str(tmp_path), "USER": "tester", **env_extra}
    return subprocess.run([shutil.which("bash"), str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


@needs_bash
def test_script_runs_orca_in_place(tmp_path):
    b = fake_orca(tmp_path, 'echo "ran $1 from $(pwd)"; echo "ORCA TERMINATED NORMALLY"\n')
    text, _ = build_script(spec(tmp_path))
    r = run_script(tmp_path, text, b, {"SLURM_SUBMIT_DIR": str(tmp_path)})
    assert r.returncode == 0, r.stderr
    out = (tmp_path / "job.out").read_text()
    assert "ran job.inp" in out and "ORCA TERMINATED NORMALLY" in out


@needs_bash
def test_script_without_orca_fails_clearly(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    text, _ = build_script(spec(tmp_path))
    # PATH holds only the empty directory: /usr/bin may contain an unrelated
    # `orca` (the GNOME screen reader on desktop Linux).
    r = run_script(tmp_path, text, empty, {"SLURM_SUBMIT_DIR": str(tmp_path)},
                   system_path=False)
    assert r.returncode == 1 and "ORCA executable not found" in r.stderr
    assert not (tmp_path / "job.out").exists()


@needs_bash
def test_scratch_copies_results_back_and_cleans_up(tmp_path):
    b = fake_orca(tmp_path, 'echo "cwd=$(pwd)"; echo "ORCA TERMINATED NORMALLY"\n'
                            'echo gbw > job.gbw; echo big > job.tmp1; cat g.xyz > seen.xyz\n')
    (tmp_path / "g.xyz").write_text("geometry\n")
    base = tmp_path / "scratch"
    text, _ = build_script(spec(tmp_path, scheduler="pbs", scratch=f"{base}",
                                copy_files=["g.xyz"]))
    r = run_script(tmp_path, text, b, {"PBS_O_WORKDIR": str(tmp_path), "PBS_JOBID": "77.srv"})
    assert r.returncode == 0, r.stderr
    out = (tmp_path / "job.out").read_text()
    assert f"cwd={base}/77.srv" in out                       # ran in scratch
    assert (tmp_path / "job.gbw").read_text() == "gbw\n"      # copied back
    assert not (tmp_path / "job.tmp1").exists()               # temp files not copied
    assert not (base / "77.srv").exists()                     # scratch removed


@needs_bash
def test_scratch_still_copies_back_when_orca_fails(tmp_path):
    b = fake_orca(tmp_path, 'echo "partial output"; exit 7\n')
    text, _ = build_script(spec(tmp_path, scratch=f"{tmp_path}/scratch"))
    r = run_script(tmp_path, text, b, {"SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_JOB_ID": "9"})
    assert r.returncode == 7                                   # ORCA's status is kept
    assert "partial output" in (tmp_path / "job.out").read_text()
    assert not (tmp_path / "scratch" / "9").exists()


def test_submit_uses_the_scheduler_command(tmp_path):
    calls = []

    class R:
        returncode, stdout, stderr = 0, "Submitted batch job 42\n", ""

    def run(cmd, cwd=None, capture_output=None, text=None):
        calls.append((cmd, cwd))
        return R()

    script = tmp_path / "job.slurm"
    script.write_text("x")
    code, msg = submit(script, "slurm", run=run, which=lambda c: f"/usr/bin/{c}")
    assert code == 0 and "42" in msg and calls == [(["/usr/bin/sbatch", "job.slurm"], tmp_path)]
    with pytest.raises(RuntimeError, match="qsub not found"):
        submit(script, "pbs", run=run, which=lambda c: None)
