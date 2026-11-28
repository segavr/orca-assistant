"""Find the quantum-chemistry ORCA among the programs called `orca` in PATH.

On desktop Linux `/usr/bin/orca` is often the GNOME screen reader. The real
ORCA is recognised by the helper programs (`orca_*`) installed next to it.
`ORCA_ASSISTANT_ORCA=/path/to/orca` overrides the search. Standard library only.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_OVERRIDE = "ORCA_ASSISTANT_ORCA"
_NAMES = ("orca", "orca.exe")
# Seen in the real /usr/bin/orca of Linux Mint: a Python script that starts
# "#!/usr/bin/python3", is headed "The Orca Team" and does `from orca import ...`.
_SCREEN_READER = re.compile(r"^(?:from|import)\s+orca\b|The Orca Team", re.M)


@dataclass
class OrcaSearch:
    path: str | None
    verdict: str                  # real | unconfirmed | user | none
    detail: str = ""
    skipped: list[str] = field(default_factory=list)
    problem: str | None = None


def _is_screen_reader(path: Path) -> bool:
    """A script that imports the `orca` Python package: the GNOME screen reader."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(20000)
    except OSError:
        return False
    return head.startswith(b"#!") and bool(_SCREEN_READER.search(head.decode("utf-8", "replace")))


def classify(path: Path) -> tuple[str, str]:
    """('real' | 'unconfirmed' | 'screen_reader', reason)."""
    real = path.resolve()
    try:
        helpers = [p for p in real.parent.iterdir() if p.name.lower().startswith("orca_")]
    except OSError:
        helpers = []
    if helpers:
        return "real", f"{len(helpers)} orca_* programs next to it"
    if _is_screen_reader(real):
        return "screen_reader", ("the GNOME screen reader (a Python script that imports the "
                                 "orca package), not quantum-chemistry ORCA")
    return "unconfirmed", "no orca_* programs next to it, so it could not be confirmed"


def find_orca(environ=None) -> OrcaSearch:
    env = os.environ if environ is None else environ
    override = env.get(ENV_OVERRIDE)
    if override:
        p = Path(override).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return OrcaSearch(str(p), "user", f"set by {ENV_OVERRIDE}")
        return OrcaSearch(None, "none", problem=(
            f"{ENV_OVERRIDE} points to {override}, which is not an executable file"))

    seen: set[Path] = set()
    found: list[tuple[Path, str, str]] = []
    for d in env.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        for name in _NAMES:
            p = Path(d) / name
            if p.is_file() and os.access(p, os.X_OK):
                key = p.resolve()
                if key not in seen:
                    seen.add(key)
                    verdict, why = classify(p)
                    found.append((p, verdict, why))

    # PATH order is respected: the first program that is not the screen reader wins.
    chosen = next((f for f in found if f[1] != "screen_reader"), None)
    skipped = []
    for p, verdict, why in found:
        if chosen and p == chosen[0]:
            continue
        if verdict == "real":
            skipped.append(f"{p}: another ORCA later in PATH, not used")
        elif verdict == "unconfirmed":
            skipped.append(f"{p}: another 'orca' later in PATH, not used")
        else:
            skipped.append(f"{p}: {why}")
    if chosen:
        return OrcaSearch(str(chosen[0]), chosen[1], chosen[2], skipped)
    return OrcaSearch(None, "none", skipped=skipped)
