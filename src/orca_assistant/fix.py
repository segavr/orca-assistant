"""Build a corrected ORCA input from a diagnosis. Standard library only.

Only changes that do not alter the physics are applied automatically
(convergence aids, more optimisation cycles, restart geometry). Changing the
spin state needs an explicit request. Everything else is reported, not guessed.
The functions are pure: the original input is never modified.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .diagnose import _MULT, diagnose
from .extract import parse_geometries

SCF_MAXITER = 500
OPT_MAXITER_MIN = 200

# Common names only; this is not a list of everything ORCA accepts.
COMMON_KEYWORDS = [
    "def2-SV(P)", "def2-SVP", "def2-SVPD", "def2-TZVP", "def2-TZVPP", "def2-TZVPD",
    "def2-TZVPPD", "def2-QZVP", "def2-QZVPP", "def2-QZVPD", "def2-QZVPPD",
    "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ",
    "6-31G", "6-31G*", "6-31+G*", "6-311G**", "6-311+G**", "STO-3G",
    "B3LYP", "PBE0", "PBE", "BP86", "TPSS", "TPSSh", "M06-2X", "CAM-B3LYP", "wB97X-D3",
    "wB97X-D3BJ", "wB97X-V", "wB97M-V", "r2SCAN-3c", "B97-3c", "PBEh-3c", "HF-3c",
    "HF", "MP2", "CCSD(T)", "DLPNO-CCSD(T)", "D3BJ", "D3ZERO", "D4",
    "Opt", "TightOpt", "OptTS", "Freq", "NumFreq", "TightSCF", "SlowConv",
    "VerySlowConv", "RIJCOSX", "RI",
]

_BANG = re.compile(r"^![^\n]*", re.M)
_XYZ_BLOCK = re.compile(
    r"^(\*[ \t]*xyz[ \t]+)(-?\d+)[ \t]+(\d+)([ \t]*\n)(.*?)\n(\*[ \t]*)$",
    re.M | re.S | re.I)


@dataclass
class FixResult:
    text: str | None = None                       # new input, None if nothing was changed
    changes: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    diff: str = ""


def _parse_xyz(text: str):
    lines = [l for l in text.splitlines() if l.strip()]
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if len(p) < 4:
            break
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def _set_option(inp: str, block: str, key: str, target) -> tuple[str, str | None]:
    """Raise `key` inside `%block`. `target(old_or_None)` returns the wanted value.
    Returns (new text, description or None if nothing changed)."""
    pat = re.compile(rf"(%[ \t]*{block}\b)(.*?)(\bend\b)", re.I | re.S)
    m = pat.search(inp)
    if m:
        body = m.group(2)
        k = re.search(rf"\b{key}[ \t]+(\d+)", body, re.I)
        if k:
            old = int(k.group(1))
            new = target(old)
            if new == old:
                return inp, None
            start = m.start(2) + k.start(1)
            return (inp[:start] + str(new) + inp[m.start(2) + k.end(1):],
                    f"%{block} {key}: {old} -> {new}")
        new = target(None)
        return (inp[:m.end(1)] + f" {key} {new}" + inp[m.end(1):],
                f"%{block}: added {key} {new}")
    new = target(None)
    star = re.search(r"^\*", inp, re.M)
    if not star:
        return inp, None
    return (inp[:star.start()] + f"%{block} {key} {new} end\n" + inp[star.start():],
            f"added a %{block} block with {key} {new}")


def _fix_scf(inp: str, res: FixResult) -> str:
    if re.search(r"\bVerySlowConv\b", inp, re.I):
        res.skipped.append("SCF: VerySlowConv is already set; check the geometry, charge "
                           "and multiplicity instead (a very stretched or wrong structure "
                           "will not converge).")
    elif re.search(r"\bSlowConv\b", inp, re.I):
        inp = re.sub(r"\bSlowConv\b", "VerySlowConv", inp, count=1, flags=re.I)
        res.changes.append("! line: SlowConv -> VerySlowConv")
    else:
        m = _BANG.search(inp)
        if m:
            inp = inp[:m.end()] + " SlowConv" + inp[m.end():]
            res.changes.append("! line: added SlowConv")
        else:
            res.skipped.append("SCF: no '!' line found to add SlowConv to.")
    inp, change = _set_option(inp, "scf", "MaxIter",
                              lambda old: SCF_MAXITER if old is None else max(old, SCF_MAXITER))
    if change:
        res.changes.append(change)
    return inp


def _first_token_element(line: str) -> str:
    m = re.match(r"\s*([A-Za-z]{1,2})", line)
    return m.group(1).capitalize() if m else ""


def _fix_opt(inp: str, out_text: str, geometry_text: str | None, res: FixResult) -> str:
    m = _XYZ_BLOCK.search(inp)
    if not m:
        res.skipped.append("Optimisation: the input has no inline '* xyz' coordinate block, "
                           "so a restart geometry cannot be written.")
        return inp
    old_lines = [l for l in m.group(5).splitlines() if l.strip()]
    if geometry_text is not None:
        new_atoms, source = _parse_xyz(geometry_text), "the given xyz file"
    else:
        geoms = parse_geometries(out_text)
        new_atoms = geoms[-1] if geoms else []
        source = "the last geometry printed in the output"
    if not new_atoms:
        res.skipped.append("Optimisation: no geometry found to restart from.")
        return inp
    if len(new_atoms) != len(old_lines):
        res.skipped.append(f"Optimisation: the input has {len(old_lines)} atoms but the "
                           f"geometry has {len(new_atoms)}; not replacing coordinates.")
        return inp
    for old, new in zip(old_lines, new_atoms):
        if _first_token_element(old) != new[0].capitalize():
            res.skipped.append("Optimisation: element order differs between input and "
                               "geometry; not replacing coordinates.")
            return inp
    # keep the original atom labels (first token) and replace only the numbers
    body = "\n".join(f"{old.split()[0]:<2} {x:14.8f} {y:14.8f} {z:14.8f}"
                     for old, (_, x, y, z) in zip(old_lines, new_atoms))
    inp = inp[:m.start(5)] + body + inp[m.end(5):]
    res.changes.append(f"coordinates replaced by {source}")
    inp, change = _set_option(
        inp, "geom", "MaxIter",
        lambda old: OPT_MAXITER_MIN if old is None else max(OPT_MAXITER_MIN, 2 * old))
    if change:
        res.changes.append(change)
    return inp


def _valid_multiplicities(electrons: int) -> list[int]:
    return [k for k in range(1, min(electrons + 1, 9) + 1) if (k - 1) % 2 == electrons % 2]


def _fix_multiplicity(inp: str, out_text: str, wanted: int | None, res: FixResult) -> str:
    m = _MULT.search(out_text)
    electrons = int(m.group(2)) if m else None
    valid = _valid_multiplicities(electrons) if electrons else []
    if wanted is None:
        res.skipped.append(
            "Multiplicity: not changed automatically, because it sets the spin state. "
            f"Valid values for {electrons} electrons: {', '.join(map(str, valid))}. "
            "Rerun with --multiplicity N (and check the charge).")
        return inp
    if wanted not in valid:
        res.skipped.append(f"Multiplicity {wanted} is impossible for {electrons} electrons; "
                           f"valid values: {', '.join(map(str, valid))}.")
        return inp
    b = _XYZ_BLOCK.search(inp)
    if not b:
        res.skipped.append("Multiplicity: no inline '* xyz charge mult' line found to edit.")
        return inp
    old = int(b.group(3))
    inp = inp[:b.start(3)] + str(wanted) + inp[b.end(3):]
    res.changes.append(f"multiplicity: {old} -> {wanted} (charge {b.group(2)} unchanged)")
    return inp


def _suggest_keywords(detail: str) -> str:
    lookup = {k.lower(): k for k in COMMON_KEYWORDS}
    parts = []
    for word in detail.split():
        close = difflib.get_close_matches(word.lower(), list(lookup), n=3, cutoff=0.75)
        if close:
            parts.append(f"{word}: did you mean {' or '.join(lookup[c] for c in close)}?")
        else:
            parts.append(f"{word}: no similar common keyword found")
    return "; ".join(parts)


def make_fix(inp_text: str, out_text: str, multiplicity: int | None = None,
             geometry_text: str | None = None) -> FixResult:
    res = FixResult()
    text = inp_text
    findings = {f.code: f for f in diagnose(out_text)}

    if "scf_not_converged" in findings:
        text = _fix_scf(text, res)
    if "opt_not_converged" in findings:
        text = _fix_opt(text, out_text, geometry_text, res)
    if "bad_multiplicity" in findings:
        text = _fix_multiplicity(text, out_text, multiplicity, res)
    elif multiplicity is not None:
        res.skipped.append("--multiplicity ignored: ORCA reported no charge/multiplicity "
                           "error in this output.")
    if "unknown_keyword" in findings:
        res.skipped.append("Unknown keyword: not changed. "
                           + _suggest_keywords(findings["unknown_keyword"].detail)
                           + " (suggestions come from a list of common names).")
    for code, why in (
            ("imaginary_frequencies", "Imaginary frequencies: not automated yet (needs a "
                                      "displacement along the normal mode)."),
            ("ts_imaginary_count", "Transition-state frequency count: not automated."),
            ("error_termination", "ORCA error without a specific rule: not automated."),
            ("truncated_output", "Output ends abruptly: nothing to fix in the input."),
            ("input_error", "Input error: fix it by hand as described by ORCA.")):
        if code in findings:
            res.skipped.append(why)

    if res.changes and text != inp_text:
        res.text = text
        res.diff = "".join(difflib.unified_diff(
            inp_text.splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile="original", tofile="fixed"))
    else:
        res.changes = []
    return res


def next_fix_path(inp: Path) -> Path:
    n = 1
    while True:
        p = inp.with_name(f"{inp.stem}_fix{n}.inp")
        if not p.exists():
            return p
        n += 1
