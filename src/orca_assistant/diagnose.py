"""Rule-based diagnosis of ORCA output text. Standard library only.

Rules look at the text of the output, not at the process exit code: ORCA can
stop with an error and still return 0 (seen with a non-converged SCF). Message
formats were taken from real ORCA 6.1.1 outputs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_BANG_LINE = re.compile(r"^\s*!{5,}\s*$")
_MULT = re.compile(
    r"Error : multiplicity \((\d+)\) is (?:even|odd) and number of electrons "
    r"\((\d+)\) is (?:even|odd) -> impossible")
_SCF_FAIL = re.compile(r"SCF NOT CONVERGED AFTER\s+(\d+)\s+CYCLES")
_SCF_FAIL_MSG = re.compile(r"the SCF has not converged")
_OPT_MAX = re.compile(
    r"The optimization did not converge but reached the maximum number of\s+"
    r"optimization cycles")
_ERR_TERM = re.compile(r"ORCA finished by error termination in (\S+)")
_FILE_MSG = re.compile(r"^\[file [^\]]*\]:[ \t]*(\S.*)$", re.M)
_FREQ = re.compile(
    r"^\s*(\d+):\s+(-?\d+\.\d+)\s+cm\*\*-1(\s+\*\*\*imaginary mode\*\*\*)?\s*$")
_TS_INPUT = re.compile(r"^\|\s*\d+>\s*!.*\b(?:OptTS|NEB-TS|ScanTS)\b", re.I | re.M)

NORMAL = "ORCA TERMINATED NORMALLY"


@dataclass
class Finding:
    code: str
    severity: str        # "error" | "warning"
    title: str
    detail: str
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


def _input_error(text: str) -> Finding | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "INPUT ERROR":
            body = []
            for nxt in lines[i + 1:]:
                if _BANG_LINE.match(nxt):
                    break
                if nxt.strip():
                    body.append(nxt.strip())
            if body and body[0].startswith("UNRECOGNIZED OR DUPLICATED KEYWORD"):
                words = " ".join(body[1:]) or "(not listed)"
                return Finding(
                    "unknown_keyword", "error", "Unrecognised or duplicated keyword in the ! line",
                    words,
                    "Check the spelling of the listed keyword(s) (basis set, functional, "
                    "job type) and remove duplicates; keywords are not case sensitive.")
            return Finding("input_error", "error", "ORCA rejected the input",
                           " ".join(body) or "(no details)",
                           "Fix the input as described in the message above and run again.")
    return None


def _multiplicity(text: str) -> Finding | None:
    m = _MULT.search(text)
    if not m:
        return None
    mult, electrons = int(m.group(1)), int(m.group(2))
    valid = [k for k in range(1, min(electrons + 1, 7) + 1)
             if (k - 1) % 2 == electrons % 2]
    return Finding(
        "bad_multiplicity", "error", "Charge and multiplicity do not match",
        f"multiplicity {mult} with {electrons} electrons is impossible",
        "For this number of electrons the multiplicity must be "
        + ", ".join(map(str, valid)) + " (or higher with the same parity). "
        "Check the charge too: changing it changes the electron count.")


def _scf(text: str, terminated_ok: bool) -> Finding | None:
    m = _SCF_FAIL.search(text)
    if not m and not _SCF_FAIL_MSG.search(text):
        return None
    detail = f"SCF not converged after {m.group(1)} cycles" if m else "SCF has not converged"
    return Finding(
        "scf_not_converged", "warning" if terminated_ok else "error",
        "SCF did not converge", detail,
        "Possible remedies: raise the iteration limit (%scf MaxIter 500 end), add "
        "SlowConv (or VerySlowConv) to the ! line, check the geometry (bond lengths, "
        "units), charge and multiplicity, or start from a converged calculation of a "
        "similar system (MORead).")


def _opt_limit(text: str) -> Finding | None:
    if not _OPT_MAX.search(text):
        return None
    return Finding(
        "opt_not_converged", "warning",
        "Geometry optimisation stopped at the maximum number of cycles",
        "ORCA reports that the optimisation did not converge",
        "Restart from the last geometry (orca-assistant xyz file.out) with more cycles "
        "(%geom MaxIter 200 end); look at the energy trend (orca-assistant plot opt "
        "file.out). If the energy keeps falling slowly or oscillates, a better starting "
        "Hessian (%geom Calc_Hess true end) often helps.")


def _frequencies(text: str) -> list[tuple[float, bool]]:
    idx = text.rfind("VIBRATIONAL FREQUENCIES")
    if idx < 0:
        return []
    modes: list[tuple[float, bool]] = []
    for line in text[idx:].splitlines()[1:]:
        m = _FREQ.match(line)
        if m:
            modes.append((float(m.group(2)), bool(m.group(3))))
        elif modes:
            break
    return modes


def _imaginary(text: str) -> Finding | None:
    modes = _frequencies(text)
    if not modes:
        return None
    imag = [f for f, flag in modes if flag]
    ts = bool(_TS_INPUT.search(text))
    if ts and len(imag) == 1:
        return None
    if not ts and not imag:
        return None
    listed = ", ".join(f"{f:.2f}" for f in imag) or "none"
    if ts:
        return Finding(
            "ts_imaginary_count", "warning",
            "Transition-state search needs exactly one imaginary frequency",
            f"{len(imag)} imaginary frequencies: {listed} cm-1",
            "With none, the optimisation probably ended in a minimum or the initial "
            "Hessian misses the reaction mode; with several, the structure is a higher-"
            "order saddle point. Check that the single mode connects the intended minima "
            "(IRC).")
    if max(abs(f) for f in imag) < 100:
        hint = ("Small imaginary frequencies are often numerical noise or a loosely "
                "converged geometry: re-optimise with TightOpt and a finer grid "
                "(DefGrid3), then repeat the frequency calculation.")
    else:
        hint = ("The structure is a saddle point, not a minimum. Displace the atoms "
                "along the imaginary mode and re-optimise. Equal imaginary values "
                "(a degenerate pair) often mean a start geometry of higher symmetry than "
                "the real minimum, for example a linear starting structure of a bent "
                "molecule.")
    return Finding("imaginary_frequencies", "warning",
                   f"{len(imag)} imaginary frequenc{'y' if len(imag) == 1 else 'ies'} "
                   "in a minimum search", f"{listed} cm-1", hint)


def diagnose(text: str) -> list[Finding]:
    terminated_ok = NORMAL in text
    specific = [f for f in (_input_error(text), _multiplicity(text),
                            _scf(text, terminated_ok), _opt_limit(text),
                            _imaginary(text)) if f]

    if not any(f.severity == "error" for f in specific):
        err = _ERR_TERM.search(text)
        if err:
            msgs = [m.group(1).strip() for m in _FILE_MSG.finditer(text)]
            specific.append(Finding(
                "error_termination", "error",
                f"ORCA stopped with an error in {err.group(1)}",
                msgs[-1] if msgs else "no message printed",
                "Read the last lines of the output for the reason; this error has no "
                "specific rule yet."))
        elif not terminated_ok:
            if not text.strip():
                specific.append(Finding(
                    "empty_output", "error", "The output file is empty",
                    "no text", "ORCA did not start or the file was truncated; check the "
                    "input path and the job's log."))
            else:
                specific.append(Finding(
                    "truncated_output", "error",
                    "The output ends without a termination message",
                    "neither normal termination nor an ORCA error message",
                    "Either the job is still running (orca-assistant watch file.out) or it "
                    "was killed, for example by a time limit or, possibly, by running out "
                    "of memory (this is a guess: ORCA prints nothing in that case). Check "
                    "the scheduler log and reduce nprocs x maxcore if memory is the "
                    "suspect."))

    order = {"error": 0, "warning": 1}
    return sorted(specific, key=lambda f: order[f.severity])


def format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "no problems found\n"
    out = []
    for f in findings:
        out.append(f"[{f.severity.upper()}] {f.title}")
        out.append(f"  evidence  : {f.detail}")
        out.append(f"  suggestion: {f.suggestion}")
        out.append("")
    return "\n".join(out)
