"""Environment report for `orca-assistant doctor`. Probes are injected so the
report can be tested without the real machine. Standard library only."""
from __future__ import annotations

from dataclasses import dataclass

from .orcafind import ENV_OVERRIDE, OrcaSearch
from .resources import Hardware, suggest_resources


@dataclass
class Report:
    text: str
    ok: bool            # ORCA was found


def _gib(mb: int | None) -> str:
    return "unknown" if mb is None else f"{mb / 1024:.1f} GiB"


def run_doctor(find, hardware, which, version_of, python_version: str) -> Report:
    found: OrcaSearch = find()
    rows: list[tuple[str, str]] = []

    if found.path:
        label = {"real": "confirmed", "user": "chosen by you",
                 "unconfirmed": "NOT confirmed"}[found.verdict]
        rows.append(("ORCA", f"{found.path}  ({label}: {found.detail})"))
    else:
        rows.append(("ORCA", "NOT FOUND" + (f": {found.problem}" if found.problem else "")))
    for s in found.skipped:
        rows.append(("", f"skipped: {s}"))
    if not found.path:
        rows.append(("", f"install ORCA and add it to PATH, or set {ENV_OVERRIDE}=/full/path/orca"))

    rows.append(("Python", python_version))
    for mod, extra in (("matplotlib", "plots: pip install \"orca-assistant[plot]\""),
                       ("anthropic", "Claude backend: pip install \"orca-assistant[llm]\"")):
        v = version_of(mod)
        rows.append((mod, v if v else f"not installed (optional; {extra})"))

    hw: Hardware = hardware()
    rows.append(("cores / RAM", f"{hw.cores} cores, {_gib(hw.ram_mb)} total, "
                                f"{_gib(hw.available_mb)} available"))
    sug = suggest_resources(hw)
    rows.append(("suggested", f"nprocs={sug.nprocs}, maxcore={sug.maxcore_mb} MB per core"))

    for tool, why in (("bash", "needed to run generated batch scripts locally"),
                      ("sbatch", "SLURM, only on a cluster"), ("qsub", "PBS, only on a cluster")):
        p = which(tool)
        rows.append((tool, p if p else f"not found ({why})"))

    width = max(len(k) for k, _ in rows)
    text = "\n".join(f"{k:<{width}} : {v}" if k else f"{'':<{width}}   {v}"
                     for k, v in rows) + "\n"
    return Report(text, found.path is not None)
