"""Batch scripts for SLURM and PBS.

ORCA launches its own MPI processes from ``%pal nprocs``, so a job asks the
scheduler for N tasks on one node instead of N threads, and ``%maxcore`` is
per core, which is exactly SLURM's ``--mem-per-cpu``. The generated script
keeps those two numbers together so a job cannot silently ask for less memory
than ORCA will use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SCHEDULERS = ("slurm", "pbs")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_PARTITION = re.compile(r"^[A-Za-z0-9._-]+$")
_MODULE = re.compile(r"^[A-Za-z0-9._/+:-]+$")
# SLURM: minutes, minutes:seconds, hours:minutes:seconds, days-hours[:mm[:ss]]
_TIME = re.compile(r"^(?:\d+|\d+:\d{2}|\d+:\d{2}:\d{2}|\d+-\d+(?::\d{2}){0,2})$")


def default_job_name(input_name: str) -> str:
    stem = input_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (cleaned or "orca")[:48]


@dataclass(frozen=True)
class JobConfig:
    input_name: str
    nprocs: int = 1
    maxcore_mb: int | None = None
    job_name: str = ""
    time_limit: str = "24:00:00"
    partition: str | None = None
    modules: tuple[str, ...] = ()
    nodes: int = 1
    scheduler: str = "slurm"

    def __post_init__(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.scheduler!r}; use slurm or pbs")
        name = self.input_name.strip()
        if not name or re.search(r"\s", name):
            raise ValueError(f"input file name {self.input_name!r} must not be empty or contain spaces")
        if name.startswith("-"):
            raise ValueError(f"input file name {name!r} must not start with '-'")
        if not isinstance(self.nprocs, int) or self.nprocs < 1:
            raise ValueError(f"nprocs must be an integer >= 1, got {self.nprocs!r}")
        if self.maxcore_mb is not None and (not isinstance(self.maxcore_mb, int)
                                           or self.maxcore_mb < 1):
            raise ValueError(f"maxcore_mb must be None or an integer >= 1, "
                             f"got {self.maxcore_mb!r}")
        if not isinstance(self.nodes, int) or self.nodes < 1:
            raise ValueError(f"nodes must be an integer >= 1, got {self.nodes!r}")
        if not _TIME.match(self.time_limit.strip()):
            raise ValueError(f"time limit {self.time_limit!r} is not a SLURM/PBS time "
                             "(use e.g. 24:00:00 or 2-12:00:00)")
        job = self.job_name.strip()
        if job and not _NAME.match(job):
            raise ValueError(f"job name {job!r} may only contain letters, digits, "
                             "'.', '_', '+' and '-'")
        if self.partition is not None and not _PARTITION.match(self.partition):
            raise ValueError(f"partition {self.partition!r} may only contain letters, "
                             "digits, '.', '_' and '-'")
        for module in self.modules:
            if not _MODULE.match(module):
                raise ValueError(f"module {module!r} may only contain letters, digits "
                                 "and '.', '_', '/', '+', ':', '-'")

    @property
    def resolved_job_name(self) -> str:
        return self.job_name.strip() or default_job_name(self.input_name)

    @property
    def output_stem(self) -> str:
        return self.input_name.rsplit(".", 1)[0] if "." in self.input_name else self.input_name

    @property
    def total_memory_mb(self) -> int | None:
        """Per node: nprocs processes x maxcore MB."""
        return None if self.maxcore_mb is None else self.nprocs * self.maxcore_mb


@dataclass
class Script:
    text: str
    warnings: list[str] = field(default_factory=list)


def job_warnings(cfg: JobConfig) -> list[str]:
    notes: list[str] = []
    if cfg.maxcore_mb is None:
        notes.append("the input has no %maxcore, so the script requests no memory; "
                     "add a memory request on a cluster that requires one")
    if cfg.nprocs == 1:
        notes.append("nprocs is 1: the script asks for a single task")
    return notes


def _prelude(cfg: JobConfig) -> list[str]:
    lines = ["", "set -euo pipefail", ""]
    if cfg.modules:
        lines += ["module purge"]
        lines += [f"module load {m}" for m in cfg.modules]
        lines.append("")
    return lines


def slurm_script(cfg: JobConfig) -> Script:
    name = cfg.resolved_job_name
    head = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --nodes={cfg.nodes}",
        f"#SBATCH --ntasks-per-node={cfg.nprocs}",
        "#SBATCH --cpus-per-task=1",
    ]
    if cfg.maxcore_mb is not None:
        head.append(f"#SBATCH --mem-per-cpu={cfg.maxcore_mb}M")
    head.append(f"#SBATCH --time={cfg.time_limit.strip()}")
    if cfg.partition:
        head.append(f"#SBATCH --partition={cfg.partition}")
    head.append(f"#SBATCH --output={name}-%j.out")

    body = [
        "",
        "# Generated by orca-assistant.",
        "# ORCA starts its own MPI ranks from %pal nprocs, so --ntasks-per-node",
        f"# stays equal to nprocs ({cfg.nprocs}).",
    ]
    if cfg.maxcore_mb is None:
        body.append("# No %maxcore in the input: no memory request is made.")
    else:
        body.append("# %maxcore is per core, hence --mem-per-cpu rather than --mem.")
    body += _prelude(cfg)
    body += [
        'cd "${SLURM_SUBMIT_DIR:-$PWD}"',
        "",
        'echo "start $(date) on $(hostname), ${SLURM_NTASKS:-?} tasks"',
        f"orca {cfg.input_name} > {cfg.output_stem}.out",
        'echo "done $(date)"',
    ]
    return Script("\n".join(head + body) + "\n", job_warnings(cfg))


def pbs_script(cfg: JobConfig) -> Script:
    name = cfg.resolved_job_name
    head = [
        "#!/bin/bash",
        f"#PBS -N {name}",
        f"#PBS -l nodes={cfg.nodes}:ppn={cfg.nprocs}",
    ]
    total = cfg.total_memory_mb
    if total is not None:
        head.append(f"#PBS -l mem={total}mb")
    head.append(f"#PBS -l walltime={cfg.time_limit.strip()}")
    if cfg.partition:
        head.append(f"#PBS -q {cfg.partition}")
    head += ["#PBS -j oe", f"#PBS -o {name}.log"]

    body = [
        "",
        "# Generated by orca-assistant.",
        f"# nodes:ppn={cfg.nprocs} matches %pal nprocs ({cfg.nprocs}).",
    ]
    if cfg.maxcore_mb is None:
        body.append("# No %maxcore in the input: no memory request is made.")
    else:
        body.append("# PBS memory is per node, so mem is nprocs x maxcore in MB.")
    body += _prelude(cfg)
    body += [
        'cd "${PBS_O_WORKDIR:-$PWD}"',
        "",
        'echo "start $(date) on $(hostname)"',
        f"orca {cfg.input_name} > {cfg.output_stem}.out",
        'echo "done $(date)"',
    ]
    return Script("\n".join(head + body) + "\n", job_warnings(cfg))


def make_script(cfg: JobConfig) -> Script:
    """Render the script for ``cfg.scheduler``."""
    return slurm_script(cfg) if cfg.scheduler == "slurm" else pbs_script(cfg)
