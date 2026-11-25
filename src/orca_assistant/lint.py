"""Rule-based sanity checks. Warnings, not errors: the user decides."""
from __future__ import annotations

from dataclasses import dataclass

from .resources import Hardware, suggest_resources
from .spec import CalcSpec, Resources, Task


@dataclass
class Issue:
    level: str  # "warning" | "info"
    message: str


def _gib(mb: int) -> str:
    return f"{mb / 1024:.1f} GiB"


def check_hardware(resources: Resources, hw: Hardware) -> list[Issue]:
    """Compare the requested nprocs x maxcore with the machine."""
    issues: list[Issue] = []
    r = resources
    need = r.nprocs * r.maxcore_mb
    sug = suggest_resources(hw)
    hint = f" Suggested for this machine: nprocs={sug.nprocs}, maxcore={sug.maxcore_mb}."

    if hw.cores and r.nprocs > hw.cores:
        issues.append(Issue("warning",
            f"nprocs={r.nprocs} exceeds the {hw.cores} physical cores detected; "
            "oversubscribing the CPU usually slows ORCA down." + hint))
    if hw.ram_mb:
        if need > hw.ram_mb:
            issues.append(Issue("warning",
                f"nprocs x maxcore = {r.nprocs} x {r.maxcore_mb} MB = {_gib(need)} "
                f"exceeds the total RAM ({_gib(hw.ram_mb)}); the job will likely swap or be killed." + hint))
        elif need > 0.75 * hw.ram_mb:
            issues.append(Issue("warning",
                f"nprocs x maxcore = {_gib(need)} is over 75% of the total RAM "
                f"({_gib(hw.ram_mb)}); ORCA can exceed %maxcore, so leave headroom." + hint))
        elif hw.available_mb and need > hw.available_mb:
            issues.append(Issue("info",
                f"nprocs x maxcore = {_gib(need)} is more than the RAM available right now "
                f"({_gib(hw.available_mb)}); close other programs or lower maxcore." + hint))
    return issues


def lint(spec: CalcSpec, hardware: Hardware | None = None) -> list[Issue]:
    issues: list[Issue] = []
    m = spec.method

    if m.is_composite and (m.basis or m.dispersion):
        issues.append(Issue("warning",
            "Composite -3c method: basis/dispersion fields are ignored in the input."))
    if not m.is_composite and m.dispersion and m.carries_dispersion:
        issues.append(Issue("warning",
            f"{m.functional} already contains a dispersion correction; "
            f"{m.dispersion} is not added to the input, it would double-count it."))
    if not m.is_composite and not m.basis:
        issues.append(Issue("warning", "No basis set specified for a non-composite method."))
    if spec.task == Task.TS:
        issues.append(Issue("info",
            "TS search: verify exactly one imaginary frequency and check the mode "
            "(IRC or displacement) connects the intended minima."))
    if spec.task == Task.SCAN:
        issues.append(Issue("info",
            "Scan indices are 0-based. Re-optimise the scan maximum as a TS."))
    if len(spec.molecule.atoms) > 60 and spec.resources.nprocs < 8:
        issues.append(Issue("info",
            "Large system with few cores; consider increasing nprocs."))
    if hardware is not None:
        issues.extend(check_hardware(spec.resources, hardware))
    return issues
