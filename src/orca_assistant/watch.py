"""Live status of an ORCA job, read from its (possibly still growing) output.
Standard library only. Remaining time is deliberately not estimated: the number
of optimisation cycles is unknown in advance, so only time per cycle is shown."""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_NUM = r"-?\d+\.\d+(?:[eE][-+]?\d+)?"
_VERSION = re.compile(r"Program Version\s+(\S+)")
_CYCLE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
_FSPE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_SCF_END = re.compile(r"SCF (?:NOT )?CONVERGED AFTER")
_SCF_ITER = re.compile(r"^\s*(\d+)\s+(-\d+\.\d{8,})\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s", re.M)
_CONV = re.compile(
    rf"^\s*(Energy change|RMS gradient|MAX gradient|RMS step|MAX step)\s+"
    rf"({_NUM})\s+({_NUM})\s+(YES|NO)\s*$", re.M)
_RUNTIME = re.compile(
    r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")
_ERROR = re.compile(r"ORCA finished by error termination in (\S+)")
_FATAL = re.compile(r"^\[file orca_main/[^\]]*\]:[ \t]*(.*)$", re.M)


@dataclass
class ConvItem:
    name: str
    value: float
    tolerance: float
    converged: bool


@dataclass
class Progress:
    state: str = "starting"   # waiting, starting, calculation, scf, optimisation,
                              # frequencies, finished, failed
    orca_version: str | None = None
    opt_cycle: int | None = None
    energy_eh: float | None = None
    scf_iteration: int | None = None
    scf_delta_e: float | None = None
    convergence: list[ConvItem] = field(default_factory=list)
    opt_converged: bool = False
    finished: bool = False
    failed: bool = False
    runtime_s: float | None = None
    notes: list[str] = field(default_factory=list)


def parse_progress(text: str) -> Progress:
    p = Progress()
    m = _VERSION.search(text)
    p.orca_version = m.group(1) if m else None
    cycles = _CYCLE.findall(text)
    if cycles:
        p.opt_cycle = int(cycles[-1])
    energies = _FSPE.findall(text)
    if energies:
        p.energy_eh = float(energies[-1])

    ends = list(_SCF_END.finditer(text))
    iters = list(_SCF_ITER.finditer(text, ends[-1].end() if ends else 0))
    if iters:
        p.scf_iteration = int(iters[-1].group(1))
        p.scf_delta_e = float(iters[-1].group(3))

    idx = text.rfind("Geometry convergence")
    if idx >= 0:
        seen: set[str] = set()
        for m in _CONV.finditer(text, idx):
            if m.group(1) in seen:
                break
            seen.add(m.group(1))
            p.convergence.append(ConvItem(m.group(1), float(m.group(2)),
                                          float(m.group(3)), m.group(4) == "YES"))
            if len(p.convergence) == 5:
                break

    p.opt_converged = "THE OPTIMIZATION HAS CONVERGED" in text
    p.finished = "ORCA TERMINATED NORMALLY" in text
    err = _ERROR.search(text)
    fatal = _FATAL.search(text)
    p.failed = not p.finished and (err is not None or fatal is not None
                                   or "aborting the run" in text)
    if fatal:
        msg = fatal.group(1).strip()
        if not msg:
            from .diagnose import diagnose
            errors = [f for f in diagnose(text) if f.severity == "error"]
            msg = errors[0].title if errors else "see the output file"
        p.notes.append(f"ORCA stopped: {msg}")
    if err:
        p.notes.append(f"ORCA error termination in {err.group(1)}")
    if "SCF NOT CONVERGED AFTER" in text:
        p.notes.append("SCF did not converge")
    rt = _RUNTIME.search(text)
    if rt:
        d, h, mi, s, ms = (int(x) for x in rt.groups())
        p.runtime_s = ((d * 24 + h) * 60 + mi) * 60 + s + ms / 1000

    if p.failed:
        p.state = "failed"
    elif p.finished:
        p.state = "finished"
    elif iters:
        p.state = "scf"
    elif p.opt_converged or "VIBRATIONAL FREQUENCIES" in text:
        p.state = "frequencies"
    elif p.opt_cycle is not None:
        p.state = "optimisation"
    elif p.energy_eh is not None:
        p.state = "calculation"
    return p


_LABELS = {
    "waiting": "waiting for the output file",
    "starting": "starting up",
    "calculation": "calculation running",
    "scf": "SCF iterations",
    "optimisation": "geometry optimisation",
    "frequencies": "final single point / frequencies",
    "finished": "finished normally",
    "failed": "FAILED",
}


def _dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def format_status(p: Progress) -> str:
    head = _LABELS[p.state]
    if p.state == "scf" and p.opt_converged:
        head = "final SCF after the optimisation (single point / frequencies)"
    elif p.opt_cycle is not None and p.state not in ("finished", "failed"):
        head += f", cycle {p.opt_cycle}"
    lines = [f"status       : {head}"]
    if p.energy_eh is not None:
        lines.append(f"energy       : {p.energy_eh:.8f} Eh (last completed SCF)")
    if p.state == "scf" and p.scf_iteration is not None:
        lines.append(f"SCF          : iteration {p.scf_iteration}, "
                     f"delta E {p.scf_delta_e:.2e} Eh")
    if p.convergence and p.state in ("optimisation", "scf", "frequencies"):
        lines.append("last geometry convergence check:")
        for c in p.convergence:
            lines.append(f"  {c.name:<13} {c.value:11.7f}  (tol {c.tolerance:.1e})  "
                         f"{'YES' if c.converged else 'NO'}")
    if p.opt_converged:
        lines.append("optimisation : converged")
    if p.runtime_s is not None:
        lines.append(f"run time     : {_dur(p.runtime_s)}")
    for n in p.notes:
        lines.append(f"[warning] {n}")
    return "\n".join(lines)


def _per_cycle(seen: dict[int, float]) -> float | None:
    # The first cycle seen may have started before watching began, so its
    # first-seen time is not its start; only later cycles are used.
    t = sorted(seen.values())[1:]
    if len(t) < 2:
        return None
    return (t[-1] - t[0]) / (len(t) - 1)


def watch(path: Path, interval: float = 2.0, once: bool = False, out=None,
          sleep=time.sleep, clock=time.monotonic, max_polls: int | None = None,
          tty: bool | None = None) -> int:
    """Poll `path` until ORCA finishes. Exit code: 0 normal, 1 failed,
    2 file missing (--once), 3 gave up after max_polls."""
    out = out if out is not None else sys.stdout
    if tty is None:
        tty = bool(getattr(out, "isatty", lambda: False)())
    start = clock()
    seen: dict[int, float] = {}
    last_core = None
    polls = 0
    while True:
        now = clock()
        p = parse_progress(path.read_text(errors="replace")) if path.exists() \
            else Progress(state="waiting")
        if p.opt_cycle is not None and p.opt_cycle not in seen:
            seen[p.opt_cycle] = now
        core = format_status(p)
        timing = f"watching for : {_dur(now - start)}"
        pc = _per_cycle(seen)
        if pc is not None and not (p.finished or p.failed):
            timing += f"\ntime / cycle : {_dur(pc)} (measured while watching)"
        if tty:
            out.write("\033[H\033[J" + core + "\n" + timing + "\n")
            out.flush()
        elif core != last_core:
            out.write(core + "\n" + timing + "\n\n")
            out.flush()
        last_core = core
        if p.failed:
            return 1
        if p.finished:
            return 0
        if once:
            return 2 if p.state == "waiting" else 0
        polls += 1
        if max_polls is not None and polls >= max_polls:
            return 3
        sleep(interval)
