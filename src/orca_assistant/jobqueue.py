"""Run several ORCA inputs one after another or in parallel, within the
machine's cores and memory. Standard library only.

Jobs start in the given order. A job runs alongside others only if the total
nprocs and nprocs x maxcore of everything running stays within the budget; a
job that alone exceeds the budget still runs, but alone, with a warning.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .resources import HEADROOM, Hardware

DEFAULT_MAXCORE_MB = 1024  # ORCA's default %maxcore
_NPROCS = re.compile(r"\bnprocs\s+(\d+)", re.I)
_PALN = re.compile(r"^!.*\bPAL(\d+)\b", re.I | re.M)
_MAXCORE = re.compile(r"^\s*%maxcore\s+(\d+)", re.I | re.M)
_OK = "ORCA TERMINATED NORMALLY"


@dataclass
class Job:
    inp: Path
    nprocs: int
    mem_mb: int


@dataclass
class JobResult:
    name: str
    status: str  # done, failed, skipped, error, interrupted, not started
    seconds: float | None = None
    energy_eh: float | None = None
    notes: list[str] = field(default_factory=list)


def read_inp_resources(text: str) -> tuple[int, int]:
    """(nprocs, maxcore in MB per core) requested by an ORCA input."""
    m = _NPROCS.search(text) or _PALN.search(text)
    nprocs = int(m.group(1)) if m else 1
    m = _MAXCORE.search(text)
    return max(nprocs, 1), int(m.group(1)) if m else DEFAULT_MAXCORE_MB


def budget(hw: Hardware, max_procs: int | None = None,
           max_mem_mb: int | None = None) -> tuple[int, int]:
    procs = max_procs or hw.cores or 1
    if max_mem_mb:
        return procs, max_mem_mb
    base = hw.available_mb
    if base is None and hw.ram_mb:
        base = int(hw.ram_mb * 0.8)
    return procs, (int(base * HEADROOM) if base else 10**9)


def _finished_ok(out: Path) -> bool:
    return out.exists() and _OK in out.read_text(errors="replace")


def _finish(job: Job, seconds: float, returncode) -> JobResult:
    out = job.inp.with_suffix(".out")
    text = out.read_text(errors="replace") if out.exists() else ""
    ok = _OK in text
    notes: list[str] = []
    energy = None
    try:
        from .parser import parse_file
        res = parse_file(out)
        energy = res.final_energy_eh
        notes += list(res.problems)
    except Exception:  # a broken output must not break the queue
        pass
    if not ok and returncode:
        notes.append(f"exit code {returncode}")
    return JobResult(job.inp.stem, "done" if ok else "failed", seconds, energy, notes)


def run_queue(inputs, orca: str, hw: Hardware, max_jobs: int = 1,
              max_procs: int | None = None, max_mem_mb: int | None = None,
              force: bool = False, poll: float = 1.0, log=None,
              popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic):
    """Returns (results in input order, interrupted)."""
    log = log or (lambda m: print(m, file=sys.stderr))
    max_jobs = max(1, max_jobs)
    procs_cap, mem_cap = budget(hw, max_procs, max_mem_mb)
    order = list(dict.fromkeys(Path(p) for p in inputs))
    results: dict[Path, JobResult] = {}
    warnings: dict[Path, str] = {}
    pending: list[Job] = []

    for p in order:
        if not p.exists():
            results[p] = JobResult(p.stem, "error", notes=["input file not found"])
            continue
        if not force and _finished_ok(p.with_suffix(".out")):
            results[p] = JobResult(p.stem, "skipped",
                                   notes=["already finished normally (--force to rerun)"])
            continue
        n, maxcore = read_inp_resources(p.read_text(errors="replace"))
        pending.append(Job(p, n, n * maxcore))

    running: list[tuple[Job, object, object, float]] = []
    interrupted = False
    try:
        while pending or running:
            for entry in list(running):
                job, proc, fh, t0 = entry
                if proc.poll() is not None:
                    fh.close()
                    running.remove(entry)
                    res = _finish(job, clock() - t0, proc.returncode)
                    if job.inp in warnings:
                        res.notes.insert(0, warnings[job.inp])
                    results[job.inp] = res
                    log(f"[queue] {job.inp.name}: {res.status} ({res.seconds:.0f} s)")
            while pending and len(running) < max_jobs:
                job = pending[0]
                used_p = sum(e[0].nprocs for e in running)
                used_m = sum(e[0].mem_mb for e in running)
                fits = (used_p + job.nprocs <= procs_cap
                        and used_m + job.mem_mb <= mem_cap)
                if running and not fits:
                    break
                pending.pop(0)
                if not fits:
                    warnings[job.inp] = (f"needs {job.nprocs} procs / {job.mem_mb} MB, "
                                         f"above the budget ({procs_cap} procs / {mem_cap} MB)")
                    log(f"[queue] warning: {job.inp.name} {warnings[job.inp]}; running it alone")
                try:
                    fh = open(job.inp.with_suffix(".out"), "w")
                    proc = popen([orca, job.inp.name], cwd=job.inp.parent,
                                 stdout=fh, stderr=subprocess.STDOUT)
                except OSError as exc:
                    results[job.inp] = JobResult(job.inp.stem, "error", notes=[str(exc)])
                    continue
                running.append((job, proc, fh, clock()))
                log(f"[queue] start {job.inp.name}: {job.nprocs} procs, {job.mem_mb} MB")
            if pending or running:
                sleep(poll)
    except KeyboardInterrupt:
        interrupted = True
        for job, proc, fh, t0 in running:
            proc.terminate()
        for job, proc, fh, t0 in running:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            fh.close()
            results[job.inp] = JobResult(job.inp.stem, "interrupted", clock() - t0)
        for job in pending:
            results[job.inp] = JobResult(job.inp.stem, "not started")

    return [results[p] for p in order], interrupted


def all_ok(results) -> bool:
    return all(r.status in ("done", "skipped") for r in results)


def format_results(results) -> str:
    rows = [("job", "status", "time", "energy (Eh)", "notes")]
    for r in results:
        t = "-" if r.seconds is None else f"{r.seconds:.0f} s"
        e = "-" if r.energy_eh is None else f"{r.energy_eh:.8f}"
        rows.append((r.name, r.status, t, e, "; ".join(r.notes) or "-"))
    w = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(row[i].ljust(w[i]) for i in range(4)) + "  " + row[4] for row in rows]
    return "\n".join(lines) + "\n"
