"""Typed calculation spec: the contract between the (future) LLM layer and the
deterministic generator. The LLM only ever produces a CalcSpec; it never writes
raw ORCA syntax."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr"
).split()
ATOMIC_NUMBER = {s: i + 1 for i, s in enumerate(_SYMBOLS)}


class Task(str, Enum):
    SP = "sp"              # single point
    OPT = "opt"            # geometry optimisation
    FREQ = "freq"          # frequencies only
    OPT_FREQ = "opt_freq"  # optimisation + frequencies
    SCAN = "scan"          # relaxed surface scan
    TS = "ts"              # transition-state optimisation + frequencies


class Atom(BaseModel):
    element: str
    x: float
    y: float
    z: float

    @field_validator("element")
    @classmethod
    def _known_element(cls, v: str) -> str:
        v = v.strip().capitalize()
        if v not in ATOMIC_NUMBER:
            raise ValueError(f"unsupported element symbol: {v!r}")
        return v


def _atoms_from_rows(rows: list[tuple[int, str]]) -> list[Atom]:
    """Atom lines ('element x y z'); every error names the source line."""
    atoms: list[Atom] = []
    for lineno, line in rows:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"XYZ line {lineno}: expected 'element x y z', got {line.strip()!r}")
        element = parts[0].strip().capitalize()
        if element not in ATOMIC_NUMBER:
            raise ValueError(f"XYZ line {lineno}: unknown element symbol {parts[0]!r}")
        try:
            x, y, z = (float(v) for v in parts[1:4])
        except ValueError:
            raise ValueError(
                f"XYZ line {lineno}: coordinates must be numbers, got {line.strip()!r}"
            ) from None
        atoms.append(Atom(element=element, x=x, y=y, z=z))
    return atoms


def atoms_from_xyz(text: str) -> list[Atom]:
    """Parse standard XYZ (count, comment, atom lines) or bare atom lines.

    The count line is checked against the coordinate lines: a truncated or
    padded file raises instead of silently producing a smaller molecule. A
    snippet written without the comment line is accepted when the standard
    reading does not match the declared count.
    """
    rows = [(n, ln) for n, ln in enumerate(text.splitlines(), 1) if ln.strip()]
    if not rows:
        raise ValueError("no atoms in the XYZ input")
    head = rows[0][1].split()
    if not (len(head) == 1 and head[0].isdigit()):
        return _atoms_from_rows(rows)  # bare atom lines, no header

    declared = int(head[0])
    if declared < 1:
        raise ValueError(f"XYZ header declares {declared} atoms; at least one is required")
    body = rows[1:]
    if not body:
        raise ValueError(
            f"XYZ header declares {declared} atoms but the file has no coordinate lines")

    errors: list[ValueError] = []

    def read(candidate: list[tuple[int, str]]) -> list[Atom]:
        try:
            return _atoms_from_rows(candidate)
        except ValueError as exc:
            errors.append(exc)
            return []

    standard = read(body[1:])  # the line after the count is a free-text comment
    if standard and len(standard) == declared:
        return standard
    without_comment = read(body)
    if without_comment and len(without_comment) == declared:
        return without_comment
    found = len(standard) or len(without_comment)
    if found:
        raise ValueError(
            f"XYZ header declares {declared} atoms but the coordinate lines give {found}")
    raise errors[0]


class Molecule(BaseModel):
    atoms: list[Atom] = Field(min_length=1)
    charge: int = 0
    multiplicity: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _check_parity(self) -> "Molecule":
        n_el = sum(ATOMIC_NUMBER[a.element] for a in self.atoms) - self.charge
        if n_el <= 0:
            raise ValueError("charge removes all electrons")
        # even electron count needs odd multiplicity, and vice versa
        if n_el % 2 == self.multiplicity % 2:
            raise ValueError(
                f"{n_el} electrons is incompatible with multiplicity "
                f"{self.multiplicity}"
            )
        return self

    @classmethod
    def from_xyz(cls, text: str, charge: int = 0, multiplicity: int = 1) -> "Molecule":
        return cls(atoms=atoms_from_xyz(text), charge=charge, multiplicity=multiplicity)


class Method(BaseModel):
    # Composite "-3c" methods carry their own basis/dispersion: leave basis None.
    functional: str = "B3LYP"
    basis: str | None = "def2-SVP"
    dispersion: Literal["D3BJ", "D3ZERO", "D4"] | None = "D3BJ"
    solvent: str | None = None  # CPCM solvent name, e.g. "water"

    @property
    def is_composite(self) -> bool:
        return self.functional.lower().endswith("-3c")

    @property
    def carries_dispersion(self) -> bool:
        """True when the functional name already encodes a D3/D4 correction."""
        name = self.functional.lower()
        return "-d3" in name or "-d4" in name


class ScanSpec(BaseModel):
    kind: Literal["B", "A", "D"]       # bond / angle / dihedral
    atoms: list[int]                   # 0-based atom indices (ORCA convention)
    start: float
    end: float
    steps: int = Field(ge=2)

    @model_validator(mode="after")
    def _check_atoms(self) -> "ScanSpec":
        need = {"B": 2, "A": 3, "D": 4}[self.kind]
        if len(self.atoms) != need:
            raise ValueError(f"scan kind {self.kind} needs {need} atom indices")
        return self


class Resources(BaseModel):
    nprocs: int = Field(4, ge=1)
    maxcore_mb: int = Field(2000, ge=200)  # per core, as in ORCA's %maxcore


class CalcSpec(BaseModel):
    molecule: Molecule
    task: Task = Task.OPT_FREQ
    method: Method = Field(default_factory=Method)
    resources: Resources = Field(default_factory=Resources)
    scan: ScanSpec | None = None
    tight_scf: bool = True

    @model_validator(mode="after")
    def _scan_consistency(self) -> "CalcSpec":
        if self.task == Task.SCAN and self.scan is None:
            raise ValueError("task 'scan' requires a scan specification")
        if self.task != Task.SCAN and self.scan is not None:
            raise ValueError("scan specification given but task is not 'scan'")
        n = len(self.molecule.atoms)
        if self.scan and any(i < 0 or i >= n for i in self.scan.atoms):
            raise ValueError("scan atom index out of range (0-based)")
        return self
