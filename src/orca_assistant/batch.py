"""Batch-script generator for SLURM and PBS Pro / OpenPBS. Standard library only.

The generated script asks the scheduler for the cores and memory the input
itself requests, calls ORCA by its full path (required for parallel runs) and
can work in a scratch directory with copy-back. Torque/Maui use a different
resource syntax and are not supported.
"""
from __future__ import annotations

import math
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .jobqueue import read_inp_resources

MEM_FACTOR = 1.3      # %maxcore is a soft per-core limit; real use can exceed it
DEFAULT_TIME = "24:00:00"
SCHEDULERS = ("slurm", "pbs")
SUBMIT_COMMAND = {"slurm": "sbatch", "pbs": "qsub"}
SUFFIX = {"slurm": ".slurm", "pbs": ".pbs"}

_WORD = re.compile(r"^[A-Za-z0-9_.:+@-]+$")
_MODULE = re.compile(r"^[A-Za-z0-9_.:+@/-]+$")
_FILE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_ABS = re.compile(r"^/[A-Za-z0-9_./+@-]+$")
_SCRATCH = re.compile(r"^[A-Za-z0-9_./${}-]+$")
_TIME = re.compile(r"^(?:(\d+)-)?(\d{1,3}):([0-5]\d):([0-5]\d)$")
_PARALLEL = re.compile(r"\bnprocs\b|\bPAL\d+\b", re.I)


@dataclass
class BatchSpec:
    scheduler: str
    inp: Path
    time: str = DEFAULT_TIME
    mem_mb: int | None = None
    partition: str | None = None     # a queue in PBS
    account: str | None = None
    modules: list[str] = field(default_factory=list)
    orca: str | None = None          # absolute path; default: `orca` found at run time
    scratch: str | None = None       # base directory, e.g. /scratch/$USER
    copy_files: list[str] = field(default_factory=list)
    name: str | None = None


def _check(value: str, pattern: re.Pattern, what: str) -> str:
    if not pattern.match(value):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _time(t: str, scheduler: str) -> str:
    m = _TIME.match(t)
    if not m:
        raise ValueError(f"invalid time {t!r}; use HH:MM:SS or D-HH:MM:SS")
    d, h, mi, s = (int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    if scheduler == "slurm":
        return f"{d}-{h:02d}:{mi:02d}:{s:02d}" if d else f"{h:02d}:{mi:02d}:{s:02d}"
    return f"{d * 24 + h:02d}:{mi:02d}:{s:02d}"


def _job_name(stem: str, scheduler: str, name: str | None) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", name or stem) or "orca"
    if scheduler == "pbs":
        if not base[0].isalpha():
            base = "j" + base
        base = base[:15]
    return base


_ORCA_CHECK = '''if [ -z "${ORCA:-}" ] || [ ! -x "$ORCA" ]; then
  echo "ORCA executable not found: use --orca /full/path/orca or --module" >&2
  exit 1
fi
case "$ORCA" in
  /*) ;;
  *) echo "ORCA must be called with its full path for parallel runs: $ORCA" >&2; exit 1 ;;
esac'''


def build_script(spec: BatchSpec) -> tuple[str, list[str]]:
    """Returns (script text, notes for the user)."""
    sched = spec.scheduler
    if sched not in SCHEDULERS:
        raise ValueError(f"unknown scheduler {sched!r}; use slurm or pbs")
    inp = Path(spec.inp)
    text = inp.read_text(errors="replace")
    nprocs, maxcore = read_inp_resources(text)
    need = nprocs * maxcore

    notes: list[str] = []
    if not _PARALLEL.search(text):
        notes.append("the input has no %pal block: requesting a single core")
    if spec.mem_mb is not None:
        if spec.mem_mb <= 0:
            raise ValueError("memory must be positive (MB)")
        mem_mb = spec.mem_mb
        if mem_mb < need:
            notes.append(f"requested memory {mem_mb} MB is below nprocs x maxcore "
                         f"({need} MB); ORCA may be killed for exceeding it")
    else:
        mem_mb = math.ceil(need * MEM_FACTOR)

    t = _time(spec.time, sched)
    name = _job_name(inp.stem, sched, spec.name)
    stem, inp_q = shlex.quote(inp.stem), shlex.quote(inp.name)
    out_q = shlex.quote(inp.stem + ".out")

    if sched == "slurm":
        head = [f"#SBATCH --job-name={name}", "#SBATCH --nodes=1",
                f"#SBATCH --ntasks={nprocs}", f"#SBATCH --mem={mem_mb}M",
                f"#SBATCH --time={t}", f"#SBATCH --output={name}.%j.log"]
        if spec.partition:
            head.append(f"#SBATCH --partition={_check(spec.partition, _WORD, 'partition')}")
        if spec.account:
            head.append(f"#SBATCH --account={_check(spec.account, _WORD, 'account')}")
        submit_dir, job_id = "${SLURM_SUBMIT_DIR:-$PWD}", "${SLURM_JOB_ID:-$$}"
    else:
        head = [f"#PBS -N {name}",
                f"#PBS -l select=1:ncpus={nprocs}:mpiprocs={nprocs}:mem={mem_mb}mb",
                f"#PBS -l walltime={t}", "#PBS -j oe", f"#PBS -o {name}.pbs.log"]
        if spec.partition:
            head.append(f"#PBS -q {_check(spec.partition, _WORD, 'queue')}")
        if spec.account:
            head.append(f"#PBS -A {_check(spec.account, _WORD, 'account')}")
        submit_dir, job_id = "${PBS_O_WORKDIR:-$PWD}", "${PBS_JOBID:-$$}"

    lines = ["#!/bin/bash", *head, "",
             f"# Generated by orca-assistant: {nprocs} cores, {maxcore} MB/core in the "
             f"input, {mem_mb} MB requested.", ""]
    for mod in spec.modules:
        lines.append(f"module load {_check(mod, _MODULE, 'module name')}")
    if spec.modules:
        lines.append("")

    if spec.orca:
        lines.append(f"ORCA={shlex.quote(_check(spec.orca, _ABS, 'ORCA path'))}")
    else:
        lines.append('ORCA="$(command -v orca || true)"')
    lines += [_ORCA_CHECK, "", "set -u", f'SUBMIT_DIR="{submit_dir}"',
              'cd "$SUBMIT_DIR" || exit 1', ""]

    if spec.scratch:
        base = _check(spec.scratch, _SCRATCH, "scratch directory")
        extra = " ".join(shlex.quote(_check(f, _FILE, "file name")) for f in spec.copy_files)
        lines += [
            f'SCRATCH="{base}/{job_id}"',
            'mkdir -p "$SCRATCH" || exit 1',
            f'cp -p {inp_q}{" " + extra if extra else ""} "$SCRATCH"/ || exit 1',
            'cd "$SCRATCH" || exit 1',
            "",
            "finish() {",
            f"  rm -f {stem}*.tmp* 2>/dev/null || true",
            f'  cp -p {stem}.* "$SUBMIT_DIR"/ 2>/dev/null || true',
            '  cd "$SUBMIT_DIR" || true',
            '  if [ -d "$SCRATCH" ] && [ "$SCRATCH" != "$SUBMIT_DIR" ]; then rm -rf "$SCRATCH"; fi',
            "}",
            "trap finish EXIT",
            "",
        ]
    elif spec.copy_files:
        notes.append("--copy-files has no effect without --scratch")

    lines.append(f'"$ORCA" {inp_q} > {out_q}')
    return "\n".join(lines) + "\n", notes


def submit(script: Path, scheduler: str, run=subprocess.run, which=shutil.which):
    """Run sbatch/qsub on the script. Returns (returncode, message)."""
    cmd = SUBMIT_COMMAND[scheduler]
    exe = which(cmd)
    if not exe:
        raise RuntimeError(f"{cmd} not found in PATH; copy the script to the cluster "
                           "login node and submit it there")
    r = run([exe, script.name], cwd=script.parent, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()
