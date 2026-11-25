"""Read a hand-written ORCA input file: simple-input line, % directives and the
coordinate block. Standard library plus pydantic for the charge/multiplicity
check.

The parser is deliberately shallow. It understands the parts ``render_input``
writes, so a file produced by this package can be read back and checked, and it
reports every other ``%`` block as not checked instead of pretending to
validate ORCA's whole input language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .lint import Issue, check_hardware
from .resources import Hardware
from .spec import ATOMIC_NUMBER, Atom, Molecule, Resources

ERROR, WARNING, INFO = "error", "warning", "info"

# Directives this module reads.
CHECKED_DIRECTIVES = ("pal", "maxcore", "geom")
# Keywords in the simple-input line that ask for a geometry path.
_TASK_KEYWORDS = {"opt", "optts", "scan"}
_DISPERSION_KEYWORDS = {"d3bj", "d3zero", "d4"}

_SIMPLE = re.compile(r"^\s*!(.*)$")
_DIRECTIVE = re.compile(r"^\s*%([A-Za-z_][A-Za-z_0-9]*)\s*(.*)$")
_SCAN_LINE = re.compile(r"^\s*([BADbad])\s+((?:\d+\s+)+)=\s*(.+?)\s*$")
_NPROCS = re.compile(r"\bnprocs\s+(\d+)", re.IGNORECASE)
_INT = re.compile(r"-?\d+")
# Blocks ORCA expects to be closed: one-liners such as %maxcore are not here.
_BLOCKS_WITH_END = {"pal", "geom", "scf", "basis", "method", "ri", "elprop", "output",
                    "mdci", "rel", "plots", "loc", "tddft", "cpcm", "casscf", "mcrpa"}


@dataclass
class ScanLine:
    """One ``B/A/D i j [k [l]] = start, end[, steps]`` line from %geom."""
    line_no: int
    kind: str
    atoms: list[int]
    values: list[float]


@dataclass
class InpFile:
    path: Path | None = None
    simple_line: str | None = None
    simple_line_no: int | None = None
    keywords: list[str] = field(default_factory=list)
    nprocs: int | None = None
    nprocs_line: int | None = None
    maxcore_mb: int | None = None
    maxcore_line: int | None = None
    charge: int | None = None
    multiplicity: int | None = None
    coordinate_line: int | None = None
    coordinate_format: str | None = None
    atoms: list[Atom] = field(default_factory=list)
    scan: list[ScanLine] = field(default_factory=list)
    blocks: dict[str, str] = field(default_factory=dict)
    unchecked: list[str] = field(default_factory=list)

    @property
    def resources(self) -> Resources | None:
        """nprocs/maxcore as ORCA will apply them; None when neither is given."""
        if self.nprocs is None and self.maxcore_mb is None:
            return None
        return Resources(nprocs=self.nprocs or 1, maxcore_mb=self.maxcore_mb or 2000)

    @property
    def molecule(self) -> Molecule | None:
        """The coordinate block as a validated Molecule, when it can be built."""
        if not self.atoms:
            return None
        charge = 0 if self.charge is None else self.charge
        multiplicity = 1 if self.multiplicity is None else self.multiplicity
        try:
            return Molecule(atoms=self.atoms, charge=charge, multiplicity=multiplicity)
        except ValidationError:
            return None

    @property
    def dispersion_keywords(self) -> list[str]:
        """Explicit D3/D4 keywords in the simple-input line."""
        return [w for w in self.keywords if w.lower() in _DISPERSION_KEYWORDS]

    @property
    def methods_with_dispersion(self) -> list[str]:
        """Keywords whose name already encodes a dispersion correction."""
        return [w for w in self.keywords
                if w.lower() not in _DISPERSION_KEYWORDS
                and ("-d3" in w.lower() or "-d4" in w.lower())]

    @property
    def composite_methods(self) -> list[str]:
        return [w for w in self.keywords if w.lower().endswith("-3c")]


@dataclass
class InpReport:
    inp: InpFile
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == WARNING]

    def exit_code(self) -> int:
        """0 clean, 1 warnings, 2 errors (anything ORCA would refuse)."""
        if self.errors:
            return 2
        return 1 if self.warnings else 0

    def to_dict(self) -> dict:
        inp = self.inp
        return {
            "path": str(inp.path) if inp.path else None,
            "simple_line": inp.simple_line,
            "keywords": inp.keywords,
            "nprocs": inp.nprocs,
            "maxcore_mb": inp.maxcore_mb,
            "charge": inp.charge,
            "multiplicity": inp.multiplicity,
            "n_atoms": len(inp.atoms),
            "coordinate_format": inp.coordinate_format,
            "scan": [{"kind": s.kind, "atoms": s.atoms, "values": s.values,
                      "line": s.line_no} for s in inp.scan],
            "blocks": sorted(inp.blocks),
            "unchecked_blocks": inp.unchecked,
            "issues": [{"level": i.level, "message": i.message} for i in self.issues],
        }


def _issue(level: str, line_no: int | None, message: str) -> Issue:
    prefix = f"line {line_no}: " if line_no is not None else ""
    return Issue(level, prefix + message)


def _parse_atoms(lines: list[tuple[int, str]], issues: list[Issue]) -> list[Atom]:
    atoms: list[Atom] = []
    for lineno, line in lines:
        parts = line.split()
        if len(parts) < 4:
            issues.append(_issue(ERROR, lineno,
                                 f"coordinate line needs 'element x y z', got {line.strip()!r}"))
            continue
        element = parts[0].strip().capitalize()
        if element not in ATOMIC_NUMBER:
            issues.append(_issue(ERROR, lineno, f"unknown element symbol {parts[0]!r}"))
            continue
        try:
            x, y, z = (float(v) for v in parts[1:4])
        except ValueError:
            issues.append(_issue(ERROR, lineno,
                                 f"coordinates must be numbers, got {line.strip()!r}"))
            continue
        atoms.append(Atom(element=element, x=x, y=y, z=z))
    return atoms


def _parse_scan_values(raw: str, line_no: int, issues: list[Issue]) -> list[float]:
    values: list[float] = []
    for token in re.split(r"[,\s]+", raw.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            issues.append(_issue(ERROR, line_no, f"scan value {token!r} is not a number"))
            return []
    return values


def _check_scan(inp: InpFile, issues: list[Issue]) -> None:
    need = {"B": 2, "A": 3, "D": 4}
    n_atoms = len(inp.atoms)
    for s in inp.scan:
        want = need[s.kind]
        if len(s.atoms) != want:
            issues.append(_issue(ERROR, s.line_no,
                                 f"scan {s.kind} needs {want} atom indices, got {len(s.atoms)}"))
            continue
        if len(set(s.atoms)) != len(s.atoms):
            issues.append(_issue(ERROR, s.line_no, f"scan {s.kind} repeats an atom index"))
        if n_atoms and any(i >= n_atoms for i in s.atoms):
            bad = ", ".join(str(i) for i in s.atoms if i >= n_atoms)
            issues.append(_issue(ERROR, s.line_no,
                                 f"scan atom index {bad} is out of range for {n_atoms} atoms "
                                 "(0-based)"))
        if len(s.values) < 2:
            issues.append(_issue(ERROR, s.line_no, "scan needs at least start and end values"))
            continue
        if len(s.values) == 2:
            issues.append(_issue(INFO, s.line_no, "scan has no step count; ORCA uses its default"))
        elif len(s.values) > 3:
            issues.append(_issue(WARNING, s.line_no,
                                 f"scan has {len(s.values)} values; start, end and steps "
                                 "are expected"))
        elif s.values[2] < 2 or s.values[2] != int(s.values[2]):
            issues.append(_issue(ERROR, s.line_no,
                                 f"scan steps must be a whole number >= 2, got {s.values[2]:g}"))
        if s.values[0] == s.values[1]:
            issues.append(_issue(WARNING, s.line_no, "scan start and end are the same value"))


def _read_directives(text: str, inp: InpFile, issues: list[Issue],
                     seen: dict[str, list[int]]) -> None:
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        d = _DIRECTIVE.match(line)
        if not d:
            continue
        name, rest = d.group(1).lower(), d.group(2).strip()
        seen.setdefault(name, []).append(i)
        body = rest[: -len("end")].strip() if rest.lower().endswith("end") else rest
        inp.blocks[name] = body
        if name not in CHECKED_DIRECTIVES and name not in inp.unchecked:
            inp.unchecked.append(name)
        if name == "pal":
            m = _NPROCS.search(body)
            if not m:
                issues.append(_issue(ERROR, i, "%pal needs 'nprocs N'"))
            else:
                inp.nprocs, inp.nprocs_line = int(m.group(1)), i
                if inp.nprocs < 1:
                    issues.append(_issue(ERROR, i, f"nprocs must be >= 1, got {inp.nprocs}"))
        elif name == "maxcore":
            m = _INT.search(body)
            if not m:
                issues.append(_issue(ERROR, i, "%maxcore needs a value in MB"))
            else:
                inp.maxcore_mb, inp.maxcore_line = int(m.group()), i
                if inp.maxcore_mb < 1:
                    issues.append(_issue(ERROR, i,
                                         f"%maxcore must be positive, got {inp.maxcore_mb}"))
        inline_closed = rest.lower().endswith("end")
        closed_later = any(ln.strip().lower() == "end" for ln in lines[i:])
        if not rest and not closed_later:
            issues.append(_issue(ERROR, i, f"%{name} is never closed with 'end'"))
        elif rest and not inline_closed and name in _BLOCKS_WITH_END and not closed_later:
            issues.append(_issue(WARNING, i, f"%{name} should be closed with 'end'"))


def _read_simple_line(text: str, inp: InpFile, issues: list[Issue]) -> None:
    for i, line in enumerate(text.splitlines(), 1):
        m = _SIMPLE.match(line)
        if m and inp.simple_line is None:
            body = m.group(1).strip()
            inp.simple_line, inp.simple_line_no = body, i
            inp.keywords = body.split()
            if not body:
                issues.append(_issue(ERROR, i, "simple-input line is empty"))


def _read_coordinates(text: str, inp: InpFile, issues: list[Issue]) -> None:
    lines = text.splitlines()
    header_at = next((i for i, ln in enumerate(lines) if ln.strip().startswith("*")), None)
    if header_at is None:
        issues.append(_issue(ERROR, None,
                             "no coordinate block ('* xyz charge multiplicity' ... '*') found"))
        return
    header_no = header_at + 1
    header = lines[header_at].strip()
    fields = header.lstrip("*").split()
    fmt = fields[0].lower() if fields else ""
    inp.coordinate_line, inp.coordinate_format = header_no, fmt or "xyz"

    body: list[tuple[int, str]] = []
    closed = False
    for j in range(header_at + 1, len(lines)):
        s = lines[j].strip()
        if s.startswith("*") and len(s.split()) == 1:
            closed = True
            break
        if s:
            body.append((j + 1, lines[j]))
    if not closed:
        issues.append(_issue(ERROR, header_no, "coordinate block is never closed; add a line with '*'"))

    if fmt.endswith("file"):
        issues.append(_issue(INFO, header_no,
                             f"coordinates are read from an external file ({fmt}); "
                             "they are not checked"))
        return
    if fmt not in ("", "xyz"):
        issues.append(_issue(INFO, header_no,
                             f"coordinate format {fmt!r} is not checked (only 'xyz')"))
        return

    numbers = fields[1:]
    if len(numbers) != 2:
        issues.append(_issue(ERROR, header_no,
                             "coordinate line needs charge and multiplicity: "
                             "'* xyz <charge> <multiplicity>'"))
    else:
        try:
            inp.charge, inp.multiplicity = int(numbers[0]), int(numbers[1])
        except ValueError:
            issues.append(_issue(ERROR, header_no,
                                 f"charge and multiplicity must be integers, "
                                 f"got {' '.join(numbers)!r}"))
        if inp.multiplicity is not None and inp.multiplicity < 1:
            issues.append(_issue(ERROR, header_no,
                                 f"multiplicity must be >= 1, got {inp.multiplicity}"))
    before = len([i for i in issues if i.level == ERROR])
    inp.atoms = _parse_atoms(body, issues)
    if not inp.atoms and len([i for i in issues if i.level == ERROR]) == before:
        issues.append(_issue(ERROR, header_no, "coordinate block has no atoms"))
    if inp.atoms and inp.charge is not None and inp.multiplicity is not None:
        try:
            Molecule(atoms=inp.atoms, charge=inp.charge, multiplicity=inp.multiplicity)
        except ValidationError as exc:
            msg = exc.errors()[0]["msg"].replace("Value error, ", "")
            issues.append(_issue(ERROR, header_no, msg))


def _read_scan(text: str, inp: InpFile, issues: list[Issue]) -> None:
    for i, line in enumerate(text.splitlines(), 1):
        m = _SCAN_LINE.match(line)
        if not m:
            continue
        values = _parse_scan_values(m.group(3), i, issues)
        inp.scan.append(ScanLine(i, m.group(1).upper(),
                                 [int(x) for x in m.group(2).split()], values))


def _cross_checks(inp: InpFile, issues: list[Issue], seen: dict[str, list[int]]) -> None:
    words = {w.lower() for w in inp.keywords}
    asked = words & _TASK_KEYWORDS
    if {"opt", "optts"} <= asked or {"optts", "scan"} <= asked:
        issues.append(_issue(WARNING, inp.simple_line_no,
                             f"contradictory task keywords: {', '.join(sorted(asked))}"))
    if "scan" in asked and not inp.scan:
        issues.append(_issue(WARNING, inp.simple_line_no,
                             "Scan is requested but no scan definition was found"))
    if inp.scan and not ({"opt", "scan"} & words):
        issues.append(_issue(INFO, inp.simple_line_no,
                             "a scan is defined but neither Opt nor Scan is in the "
                             "simple-input line; ORCA needs one of them to run it"))
    explicit = inp.dispersion_keywords
    carrying = inp.composite_methods or inp.methods_with_dispersion
    if explicit and carrying:
        issues.append(_issue(WARNING, inp.simple_line_no,
                             f"{carrying[0]} already contains a dispersion correction; "
                             f"{explicit[0]} double-counts it"))
    for name, where in (("maxcore", inp.maxcore_line), ("pal", inp.nprocs_line)):
        if len(seen.get(name, [])) > 1:
            issues.append(_issue(WARNING, where,
                                 f"%{name} is given more than once; only the last one counts"))
    for name in inp.unchecked:
        issues.append(_issue(INFO, seen[name][0],
                             f"%{name} is not checked (only %pal, %maxcore and %geom are read)"))


def parse_inp(text: str, path: Path | None = None,
              hardware: Hardware | None = None) -> InpReport:
    """Parse and check an ORCA input file. Never raises on bad input."""
    inp = InpFile(path=path)
    issues: list[Issue] = []
    seen: dict[str, list[int]] = {}
    _read_simple_line(text, inp, issues)
    if inp.simple_line is None:
        issues.append(_issue(ERROR, None, "no simple-input line ('! ...') found"))
    _read_directives(text, inp, issues, seen)
    _read_coordinates(text, inp, issues)
    _read_scan(text, inp, issues)
    _check_scan(inp, issues)
    _cross_checks(inp, issues, seen)
    if hardware is not None and inp.resources is not None:
        issues.extend(check_hardware(inp.resources, hardware))
    return InpReport(inp, issues)
