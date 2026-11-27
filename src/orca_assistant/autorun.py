"""Careful automatic restart: run ORCA, diagnose, apply only the safe fixes from
`fix`, run again, with a hard attempt limit and a journal. Standard library only.

The loop stops as soon as something needs a human decision (spin state,
keywords, imaginary frequencies, unknown errors) or when nothing can be fixed
automatically any more. The original input is never modified.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .diagnose import Finding, diagnose
from .fix import make_fix, next_fix_path

MAX_ATTEMPTS_CAP = 10
_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


@dataclass
class Attempt:
    inp: Path
    out: Path
    returncode: int | None
    seconds: float
    energy_eh: float | None
    findings: list[Finding]
    changes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    diff: str = ""
    next_inp: Path | None = None


@dataclass
class AutoResult:
    attempts: list[Attempt]
    status: str            # ok | needs_human | limit | interrupted


def default_runner(orca: str):
    """Runs ORCA on an input (cwd = its folder, output next to it)."""
    def run(inp: Path) -> int:
        out = inp.with_suffix(".out")
        with open(out, "w") as fh:
            return subprocess.run([orca, inp.name], cwd=inp.parent, stdout=fh,
                                  stderr=subprocess.STDOUT).returncode
    return run


def run_with_retries(inp: Path, runner, max_attempts: int = 3, log=None,
                     clock=time.monotonic) -> AutoResult:
    if not 1 <= max_attempts <= MAX_ATTEMPTS_CAP:
        raise ValueError(f"--max-attempts must be between 1 and {MAX_ATTEMPTS_CAP}")
    log = log or (lambda m: print(m, file=sys.stderr))
    original = Path(inp)
    current = original
    attempts: list[Attempt] = []
    status = "limit"
    try:
        while True:
            n = len(attempts) + 1
            log(f"[auto] attempt {n}/{max_attempts}: {current.name}")
            t0 = clock()
            rc = runner(current)
            seconds = clock() - t0
            out = current.with_suffix(".out")
            text = out.read_text(errors="replace") if out.exists() else ""
            energies = _ENERGY.findall(text)
            att = Attempt(current, out, rc, seconds,
                          float(energies[-1]) if energies else None, diagnose(text))
            attempts.append(att)
            if not att.findings:
                status = "ok"
                break
            if n >= max_attempts:
                status = "limit"
                break
            fix = make_fix(current.read_text(errors="replace"), text)
            att.skipped = fix.skipped
            if fix.text is None:
                status = "needs_human"
                break
            dest = next_fix_path(original)
            dest.write_text(fix.text)
            att.changes, att.diff, att.next_inp = fix.changes, fix.diff, dest
            log(f"[auto] {att.findings[0].title}; wrote {dest.name}: " + "; ".join(fix.changes))
            current = dest
    except KeyboardInterrupt:
        status = "interrupted"
    return AutoResult(attempts, status)


_VERDICT = {
    "ok": "finished without problems",
    "needs_human": "stopped: the remaining problem needs your decision",
    "limit": "stopped: problems remain and the attempt limit was reached",
    "interrupted": "interrupted",
}


def format_summary(result: AutoResult) -> str:
    lines = []
    for i, a in enumerate(result.attempts, 1):
        verdict = "ok" if not a.findings else "; ".join(f.title for f in a.findings)
        e = "-" if a.energy_eh is None else f"{a.energy_eh:.8f} Eh"
        lines.append(f"attempt {i}: {a.inp.name}  {a.seconds:.0f} s  {e}  {verdict}")
        for s in a.skipped:
            lines.append(f"    not changed automatically: {s}")
    lines.append(f"result: {_VERDICT[result.status]} ({len(result.attempts)} attempt"
                 f"{'s' if len(result.attempts) != 1 else ''})")
    return "\n".join(lines) + "\n"


def format_journal(result: AutoResult, original: Path, when: str | None = None) -> str:
    when = when or datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# Automatic run: {Path(original).name}", "",
           f"- started: {when}", f"- result: {_VERDICT[result.status]}",
           f"- attempts: {len(result.attempts)}", "",
           "| # | input | time | energy (Eh) | outcome |", "|---|---|---|---|---|"]
    for i, a in enumerate(result.attempts, 1):
        e = "-" if a.energy_eh is None else f"{a.energy_eh:.8f}"
        verdict = "ok" if not a.findings else "; ".join(f.title for f in a.findings)
        out.append(f"| {i} | {a.inp.name} | {a.seconds:.0f} s | {e} | {verdict} |")
    for i, a in enumerate(result.attempts, 1):
        out += ["", f"## Attempt {i}: {a.inp.name}", ""]
        if not a.findings:
            out.append("No problems found in the output.")
        for f in a.findings:
            out += [f"- **{f.severity.upper()}** {f.title}", f"  - evidence: {f.detail}",
                    f"  - suggestion: {f.suggestion}"]
        if a.changes:
            out += ["", f"Changes for the next attempt ({a.next_inp.name if a.next_inp else '-'}):"]
            out += [f"- {c}" for c in a.changes]
            out += ["", "```diff", a.diff.rstrip("\n"), "```"]
        if a.skipped:
            out += ["", "Not changed automatically:"] + [f"- {s}" for s in a.skipped]
    return "\n".join(out) + "\n"


def next_journal_path(inp: Path) -> Path:
    n = 1
    while True:
        p = Path(inp).with_name(f"{Path(inp).stem}_auto{n}.md")
        if not p.exists():
            return p
        n += 1
