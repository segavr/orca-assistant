"""Deterministic CalcSpec -> ORCA input text. No LLM involved."""
from __future__ import annotations

from .spec import CalcSpec, Task

# Functionals with exact exchange: use RIJCOSX to keep them affordable.
_HYBRIDS = {
    "b3lyp", "pbe0", "tpssh", "m06-2x", "cam-b3lyp",
    "wb97x-d3", "wb97x-d3bj", "wb97x-v", "wb97x-d4", "wb97m-v",
}

_TASK_KEYWORDS = {
    Task.SP: [],
    Task.OPT: ["Opt"],
    Task.FREQ: ["Freq"],
    Task.OPT_FREQ: ["Opt", "Freq"],
    Task.SCAN: ["Opt"],
    Task.TS: ["OptTS", "Freq"],
}


def _bang_line(spec: CalcSpec) -> str:
    m = spec.method
    kw: list[str] = [m.functional]
    if m.basis and not m.is_composite:
        kw.append(m.basis)
    if m.dispersion and not m.is_composite and not m.carries_dispersion:
        kw.append(m.dispersion)
    if not m.is_composite:
        kw.append("def2/J")
        kw.append("RIJCOSX" if m.functional.lower() in _HYBRIDS else "RI")
    if spec.tight_scf:
        kw.append("TightSCF")
    kw.extend(_TASK_KEYWORDS[spec.task])
    if m.solvent:
        kw.append(f"CPCM({m.solvent})")
    return "! " + " ".join(kw)


def render_input(spec: CalcSpec) -> str:
    out = [_bang_line(spec)]
    out.append(f"%pal nprocs {spec.resources.nprocs} end")
    out.append(f"%maxcore {spec.resources.maxcore_mb}")

    if spec.task == Task.SCAN and spec.scan:
        s = spec.scan
        idx = " ".join(str(i) for i in s.atoms)
        out += [
            "%geom Scan",
            f"  {s.kind} {idx} = {s.start}, {s.end}, {s.steps}",
            "  end",
            "end",
        ]
    elif spec.task == Task.TS:
        out.append("%geom Calc_Hess true end")

    mol = spec.molecule
    out.append(f"* xyz {mol.charge} {mol.multiplicity}")
    for a in mol.atoms:
        out.append(f"{a.element:<2} {a.x:>12.6f} {a.y:>12.6f} {a.z:>12.6f}")
    out.append("*")
    return "\n".join(out) + "\n"
