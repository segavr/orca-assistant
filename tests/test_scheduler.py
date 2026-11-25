import pytest

from orca_assistant.scheduler import (JobConfig, default_job_name,
                                     job_warnings, make_script, pbs_script,
                                     slurm_script)


def directives(text, prefix="#"):
    return [ln for ln in text.splitlines() if ln.startswith(prefix)]


def test_slurm_header_matches_nprocs_and_maxcore():
    text = slurm_script(JobConfig(input_name="water.inp", nprocs=6,
                                  maxcore_mb=1500)).text
    assert directives(text)[:5] == [
        "#!/bin/bash",
        "#SBATCH --job-name=water",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=6",
        "#SBATCH --cpus-per-task=1",
    ]
    assert "#SBATCH --mem-per-cpu=1500M" in directives(text)
    assert "#SBATCH --time=24:00:00" in directives(text)
    assert "orca water.inp > water.out" in text


def test_slurm_cpus_per_task_stays_one():
    """ORCA launches its own MPI ranks, so the job asks for tasks, not threads."""
    text = slurm_script(JobConfig(input_name="job.inp", nprocs=8)).text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "#SBATCH --ntasks-per-node=8" in text


def test_pbs_requests_total_memory_per_node():
    text = pbs_script(JobConfig(input_name="water.inp", nprocs=4, maxcore_mb=2000,
                                scheduler="pbs")).text
    assert "#PBS -l nodes=1:ppn=4" in text
    assert "#PBS -l mem=8000mb" in text
    assert "#PBS -l walltime=24:00:00" in text


def test_pbs_queue_flag():
    text = pbs_script(JobConfig(input_name="job.inp", nprocs=2, partition="gpu",
                                scheduler="pbs")).text
    assert "#PBS -q gpu" in text
    assert not any(line.startswith("#SBATCH") for line in text.splitlines())


def test_no_maxcore_omits_memory_request_and_warns():
    script = make_script(JobConfig(input_name="job.inp", nprocs=2))
    assert "--mem-per-cpu" not in script.text
    assert any("%maxcore" in w for w in script.warnings)


def test_modules_are_loaded_in_order():
    text = slurm_script(JobConfig(input_name="job.inp", nprocs=2,
                                  modules=("orca/6.1.1", "openmpi/5.0"))).text
    assert "module purge" in text
    assert text.index("module load orca/6.1.1") < text.index("module load openmpi/5.0")


def test_job_name_defaults_to_the_input_stem():
    assert JobConfig(input_name="benzene_opt.inp").resolved_job_name == "benzene_opt"
    assert JobConfig(input_name="job.inp", job_name="mine").resolved_job_name == "mine"


def test_default_job_name_sanitizes():
    assert default_job_name("my job!.inp") == "my_job"
    assert default_job_name(".inp") == "orca"


def test_script_is_deterministic():
    cfg = JobConfig(input_name="job.inp", nprocs=4, maxcore_mb=2000,
                    partition="cpu", modules=("orca/6.1.1",))
    assert make_script(cfg).text == make_script(cfg).text


def test_scheduler_is_selected():
    slurm = make_script(JobConfig(input_name="j.inp", scheduler="slurm")).text
    pbs = make_script(JobConfig(input_name="j.inp", scheduler="pbs")).text
    assert "#SBATCH" in slurm and "#PBS" not in slurm
    assert "#PBS" in pbs and "#SBATCH" not in pbs


def test_job_name_cannot_inject_a_directive():
    with pytest.raises(ValueError, match="job name"):
        JobConfig(input_name="j.inp", job_name="x\n#SBATCH --mem=1000G")


def test_input_name_with_a_space_is_refused():
    with pytest.raises(ValueError, match="must not be empty or contain spaces"):
        JobConfig(input_name="my job.inp")


def test_input_name_cannot_start_with_a_dash():
    with pytest.raises(ValueError, match="must not start with"):
        JobConfig(input_name="-rf.inp")


def test_nprocs_must_be_positive():
    for bad in (0, -2):
        with pytest.raises(ValueError, match="nprocs"):
            JobConfig(input_name="j.inp", nprocs=bad)


def test_maxcore_must_be_positive_or_none():
    with pytest.raises(ValueError, match="maxcore_mb"):
        JobConfig(input_name="j.inp", maxcore_mb=0)


def test_time_limit_is_validated():
    assert JobConfig(input_name="j.inp", time_limit="2-12:00:00").time_limit == "2-12:00:00"
    assert JobConfig(input_name="j.inp", time_limit="90").time_limit == "90"
    for bad in ("24h", "1:2", "-1:00:00", "24:00:00\n#SBATCH --mem=1T"):
        with pytest.raises(ValueError, match="time limit"):
            JobConfig(input_name="j.inp", time_limit=bad)


def test_partition_is_validated():
    with pytest.raises(ValueError, match="partition"):
        JobConfig(input_name="j.inp", partition="cpu\n#SBATCH --mem=1T")


def test_module_is_validated():
    with pytest.raises(ValueError, match="module"):
        JobConfig(input_name="j.inp", modules=("orca/6.1.1; rm -rf /",))


def test_nodes_must_be_positive():
    with pytest.raises(ValueError, match="nodes"):
        JobConfig(input_name="j.inp", nodes=0)


def test_warnings_mention_a_single_task():
    assert any("single task" in w for w in job_warnings(JobConfig(input_name="j.inp")))
