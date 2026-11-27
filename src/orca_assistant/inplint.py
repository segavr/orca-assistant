"""Checks for hand-written ORCA inputs, run before the calculation. Standard
library only.

Every rule is either something ORCA itself enforces (observed in real outputs:
impossible multiplicity, duplicated or unrecognised keywords) or a plain
consistency check (unclosed blocks, missing files, memory against the machine).
Rules about differences between ORCA versions are deliberately absent: they
must come from the manual, not from memory.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .fix import COMMON_KEYWORDS
from .resources import Hardware, suggest_resources

ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og").split()
Z = {sym: i for i, sym in enumerate(ELEMENTS, 1)}

# Directives that take a value and have no closing `end`.
_NO_END = {"maxcore", "base", "moinp", "id"}
_BASIS_PREFIX = re.compile(r"^(def2[-/]|cc-p|aug-cc|6-31|6-311|sto-)", re.I)
_DISPERSION = {"d3", "d3bj", "d3zero", "d4"}
DEFAULT_MAXCORE_MB = 1024      # shown as "Max core memory" default in ORCA
SCF_MAXITER_DEFAULT = 125      # printed in real outputs: "Maximum # iterations MaxIter .... 125"
GEOM_MAXITER_DEFAULT = 50      # printed in real outputs: "Max. no of cycles MaxIter .... 50"


@dataclass
class InpIssue:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str
    line: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def _block_option(text: str, block: str, key: str) -> int | None:
    m = re.search(rf"(%[ \t]*{block}\b)(.*?)(\bend\b)", text, re.I | re.S)
    if not m:
        return None
    k = re.search(rf"\b{key}[ \t]+(\d+)", m.group(2), re.I)
    return int(k.group(1)) if k else None


def _atom_electrons(token: str) -> tuple[int | None, str | None]:
    """(Z, None) for a normal atom; (None, reason) when the label is special or unknown."""
    if re.search(r"[:>(<]", token):
        return None, f"special atom label '{token}'"
    alpha = re.match(r"[A-Za-z]{1,2}", token)
    if not alpha:
        return None, f"unreadable atom label '{token}'"
    a = alpha.group(0)
    if a.capitalize() in Z and len(a) == 2:
        return Z[a.capitalize()], None
    if a[0].capitalize() in Z:
        return Z[a[0].capitalize()], None
    return None, f"unknown element '{token}'"


def _valid_multiplicities(electrons: int) -> list[int]:
    return [k for k in range(1, min(electrons + 1, 9) + 1) if (k - 1) % 2 == electrons % 2]


def _coordinates(lines: list[str], base_dir: Path | None, issues: list[InpIssue]):
    """Check the `* ...` coordinate block. Returns (charge, mult, atoms) or None."""
    head = re.compile(r"^\s*\*\s*(xyz|xyzfile|int|internal|gzmt)\b\s*(-?\d+)?\s*(\d+)?(.*)$", re.I)
    for i, raw in enumerate(lines):
        m = head.match(_strip_comment(raw))
        if not m:
            continue
        kind = m.group(1).lower()
        if m.group(2) is None or m.group(3) is None:
            issues.append(InpIssue("error", "bad_charge_mult",
                                   "the coordinate line needs a charge and a multiplicity "
                                   "(for example: * xyz 0 1)", i + 1))
            return None
        charge, mult = int(m.group(2)), int(m.group(3))
        if mult < 1:
            issues.append(InpIssue("error", "bad_charge_mult",
                                   f"multiplicity must be at least 1, got {mult}", i + 1))
        if kind == "xyzfile":
            name = m.group(4).strip().strip('"')
            if not name:
                issues.append(InpIssue("error", "missing_xyzfile",
                                       "* xyzfile line without a file name", i + 1))
            elif base_dir is not None and not (base_dir / name).exists():
                issues.append(InpIssue("error", "missing_xyzfile",
                                       f"coordinate file not found next to the input: {name}",
                                       i + 1))
            return None
        if kind != "xyz":
            return None            # internal coordinates: not checked
        atoms: list[tuple[str, int]] = []
        closed = False
        for j in range(i + 1, len(lines)):
            body = _strip_comment(lines[j]).strip()
            if re.fullmatch(r"\*|end", body, re.I):
                closed = True
                break
            if not body:
                continue
            atoms.append((body.split()[0], j + 1))
        if not closed:
            issues.append(InpIssue("error", "unclosed_coordinates",
                                   "the coordinate block is not closed with a line containing "
                                   "only '*'", i + 1))
        if not atoms:
            issues.append(InpIssue("error", "no_atoms", "the coordinate block has no atoms", i + 1))
            return None
        return charge, mult, atoms
    issues.append(InpIssue("error", "no_coordinates",
                           "no coordinate block found (a line like '* xyz 0 1' ... '*')"))
    return None


def _charge_multiplicity(coords, issues: list[InpIssue]) -> None:
    charge, mult, atoms = coords
    total, skipped = 0, []
    for label, lineno in atoms:
        z, why = _atom_electrons(label)
        if z is None:
            skipped.append(why)
        else:
            total += z
    if skipped:
        issues.append(InpIssue("info", "electron_count_skipped",
                               "multiplicity not checked against the electron count: "
                               + "; ".join(sorted(set(skipped)))))
        return
    electrons = total - charge
    if electrons < 0:
        issues.append(InpIssue("error", "bad_charge_mult",
                               f"charge {charge} leaves {electrons} electrons"))
        return
    if (mult - 1) % 2 != electrons % 2 or mult - 1 > electrons:
        valid = ", ".join(map(str, _valid_multiplicities(electrons)))
        issues.append(InpIssue(
            "error", "bad_multiplicity",
            f"multiplicity {mult} is impossible with {electrons} electrons (charge {charge}); "
            f"valid multiplicities: {valid}. ORCA stops with exactly this error."))


def _keywords(lines: list[str], issues: list[InpIssue]) -> list[str]:
    seen: dict[str, int] = {}
    tokens: list[str] = []
    lookup = {k.lower(): k for k in COMMON_KEYWORDS}
    for i, raw in enumerate(lines):
        body = _strip_comment(raw).strip()
        if not body.startswith("!"):
            continue
        for tok in body[1:].split():
            tokens.append(tok)
            low = tok.lower()
            if low in seen:
                issues.append(InpIssue(
                    "warning", "duplicate_keyword",
                    f"keyword '{tok}' appears twice (first on line {seen[low]}); ORCA "
                    "rejects duplicated keywords", i + 1))
                continue
            seen[low] = i + 1
            if low in lookup or "(" in tok or "/" in tok:
                continue
            close = difflib.get_close_matches(low, list(lookup), n=1, cutoff=0.85)
            if close:
                issues.append(InpIssue(
                    "info", "possible_typo",
                    f"'{tok}' is not in the list of common names; closest: "
                    f"{lookup[close[0]]}. Check the spelling (this list is not exhaustive).",
                    i + 1))
    if not tokens:
        issues.append(InpIssue("error", "no_keyword_line",
                               "no '!' keyword line found (for example: ! B3LYP def2-SVP Opt)"))
    return tokens


def _method(tokens: list[str], text: str, issues: list[InpIssue]) -> None:
    low = [t.lower() for t in tokens]
    if any(t.endswith("-3c") for t in low) and any(_BASIS_PREFIX.match(t) for t in low):
        issues.append(InpIssue("warning", "composite_with_basis",
                               "a composite -3c method defines its own basis set; an extra "
                               "basis keyword is ignored or may conflict"))
    embedded = [t for t in low if re.search(r"-d3|-d4", t)]
    extra = [t for t in low if t in _DISPERSION]
    if embedded and extra:
        issues.append(InpIssue("warning", "double_dispersion",
                               f"{embedded[0]} already contains a dispersion correction; also "
                               f"giving {extra[0].upper()} would count it twice"))
    if "optts" in low and not re.search(r"calc_hess|inhess|readhess|\bfreq\b", text, re.I) \
            and "freq" not in low:
        issues.append(InpIssue("info", "ts_without_hessian",
                               "OptTS needs a good initial Hessian: consider %geom Calc_Hess "
                               "true end (or a read Hessian), and run Freq on the result"))
    scf = _block_option(text, "scf", "MaxIter")
    if scf is not None and scf < 20:
        issues.append(InpIssue("warning", "low_scf_maxiter",
                               f"%scf MaxIter {scf} is far below ORCA's default "
                               f"({SCF_MAXITER_DEFAULT}); the SCF will probably stop unconverged"))
    geom = _block_option(text, "geom", "MaxIter")
    if geom is not None and geom < 10:
        issues.append(InpIssue("warning", "low_geom_maxiter",
                               f"%geom MaxIter {geom} is far below ORCA's default "
                               f"({GEOM_MAXITER_DEFAULT}); the optimisation will stop early"))


def _blocks(text: str, issues: list[InpIssue]) -> None:
    clean = "\n".join(_strip_comment(l) for l in text.splitlines())
    head = clean.split("\n*", 1)[0] if "\n*" in clean else clean
    names = [n.lower() for n in re.findall(r"%\s*(\w+)", head)]
    needs_end = [n for n in names if n not in _NO_END]
    ends = len(re.findall(r"\bend\b", head, re.I))
    if ends < len(needs_end):
        issues.append(InpIssue("warning", "unclosed_block",
                               f"{len(needs_end)} % blocks but only {ends} 'end' words before "
                               "the coordinates: check that every block is closed with 'end'"))


def _resources(text: str, hw: Hardware | None, issues: list[InpIssue]) -> None:
    pal = re.search(r"\bnprocs\s+(\d+)", text, re.I) or re.search(r"^!.*\bPAL(\d+)\b", text, re.I | re.M)
    nprocs = int(pal.group(1)) if pal else 1
    mc = re.search(r"^\s*%maxcore\s+(\d+)", text, re.I | re.M)
    maxcore = int(mc.group(1)) if mc else DEFAULT_MAXCORE_MB
    if not mc:
        issues.append(InpIssue("info", "no_maxcore",
                               f"no %maxcore: ORCA uses its default ({DEFAULT_MAXCORE_MB} MB "
                               "per core)"))
    elif maxcore < 200:
        issues.append(InpIssue("info", "tiny_maxcore",
                               f"%maxcore {maxcore} MB is unusually small (ORCA accepted it in "
                               "our tests, but typical values are 1000-4000 MB per core)"))
    if hw is None:
        return
    sug = suggest_resources(hw)
    hint = f" Suggested for this machine: nprocs={sug.nprocs}, maxcore={sug.maxcore_mb}."
    if not pal and hw.cores > 1:
        issues.append(InpIssue("info", "single_core",
                               f"no %pal block: ORCA will use 1 of the {hw.cores} cores."
                               + hint))
    if hw.cores and nprocs > hw.cores:
        issues.append(InpIssue("warning", "too_many_procs",
                               f"nprocs={nprocs} exceeds the {hw.cores} physical cores "
                               "detected; oversubscribing usually slows ORCA down." + hint))
    need = nprocs * maxcore
    if hw.ram_mb:
        gib = lambda mb: f"{mb / 1024:.1f} GiB"
        if need > hw.ram_mb:
            issues.append(InpIssue("warning", "memory_exceeds_ram",
                                   f"nprocs x maxcore = {nprocs} x {maxcore} MB = {gib(need)} "
                                   f"exceeds the total RAM ({gib(hw.ram_mb)})." + hint))
        elif need > 0.75 * hw.ram_mb:
            issues.append(InpIssue("warning", "memory_near_ram",
                                   f"nprocs x maxcore = {gib(need)} is over 75% of the total "
                                   f"RAM ({gib(hw.ram_mb)}); ORCA can exceed %maxcore." + hint))
        elif hw.available_mb and need > hw.available_mb:
            issues.append(InpIssue("info", "memory_exceeds_available",
                                   f"nprocs x maxcore = {gib(need)} is more than the RAM "
                                   f"available now ({gib(hw.available_mb)})." + hint))


def lint_input(text: str, hardware: Hardware | None = None,
               base_dir: Path | None = None) -> list[InpIssue]:
    issues: list[InpIssue] = []
    lines = text.splitlines()
    tokens = _keywords(lines, issues)
    coords = _coordinates(lines, base_dir, issues)
    if coords:
        _charge_multiplicity(coords, issues)
    _method(tokens, text, issues)
    _blocks(text, issues)
    _resources(text, hardware, issues)
    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(issues, key=lambda i: (order[i.level], i.line or 0))


def format_issues(issues: list[InpIssue]) -> str:
    if not issues:
        return "no problems found\n"
    out = []
    for i in issues:
        where = f" (line {i.line})" if i.line else ""
        out.append(f"[{i.level.upper()}] {i.message}{where}")
    return "\n".join(out) + "\n"
