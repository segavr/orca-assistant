from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from .extract import extract, parse_geometries, write_xyz
from .inp import parse_inp
from .lint import lint
from .parser import ParsedOutput, parse_file
from .render import render_input
from .resources import detect_hardware, suggest_resources
from .scheduler import JobConfig, make_script
from .spec import CalcSpec, atoms_from_xyz
from .table import build_rows, render_table
from .watch import watch


class CliError(Exception):
    """A problem with the user's input: reported as one line, not a traceback."""


def _read_text(path: Path, what: str) -> str:
    if not path.exists():
        raise CliError(f"{what} not found: {path}")
    if not path.is_file():
        raise CliError(f"{what} is not a regular file: {path}")
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise CliError(f"cannot read {what} {path}: {exc}") from None


def _validation_message(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].replace("Value error, ", "")
        lines.append(f"  {where}: {msg}" if where else f"  {msg}")
    return "\n".join(lines)


def load_spec(path: Path) -> CalcSpec:
    text = _read_text(path, "spec file")
    try:
        return CalcSpec.model_validate_json(text)
    except ValidationError as exc:
        if any(err.get("type") == "json_invalid" for err in exc.errors()):
            msg = exc.errors()[0]["msg"].replace("Invalid JSON: ", "")
            raise CliError(f"{path} is not valid JSON: {msg}") from None
        raise CliError(f"{path} is not a valid CalcSpec:\n{_validation_message(exc)}") from None
    except ValueError as exc:  # a decoder error pydantic did not wrap
        raise CliError(f"{path} is not valid JSON: {exc}") from None



def _print_lint(spec: CalcSpec) -> None:
    for issue in lint(spec, detect_hardware()):
        print(f"[{issue.level}] {issue.message}", file=sys.stderr)


def print_summary(res: ParsedOutput) -> None:
    def fmt(v, unit=""):
        return "n/a" if v is None else f"{v:.6f}{unit}"

    print(f"terminated normally : {res.terminated_normally}")
    opt = "n/a (no optimisation)" if res.opt_converged is None else str(res.opt_converged)
    print(f"optimisation        : {opt}")
    print(f"final energy        : {fmt(res.final_energy_eh, ' Eh')}")
    if res.frequencies_cm1:
        print(f"frequencies (cm-1)  : {', '.join(f'{f:.1f}' for f in res.frequencies_cm1)}")
        print(f"imaginary           : {res.n_imaginary}")
    print(f"ZPE                 : {fmt(res.zpe_eh, ' Eh')}")
    print(f"enthalpy            : {fmt(res.enthalpy_eh, ' Eh')}")
    print(f"Gibbs free energy   : {fmt(res.gibbs_eh, ' Eh')}")
    if res.runtime_s is not None:
        print(f"run time            : {res.runtime_s:.1f} s")
    for w in res.warnings:
        print(f"[warning] {w}")
    for pr in res.problems:
        print(f"[PROBLEM] {pr}")


def print_extras(ex) -> None:
    def chg(items):
        return ", ".join(f"{c.element}{c.index} {c.charge:+.3f}" for c in items)

    if ex.orca_version:
        print(f"ORCA version        : {ex.orca_version}")
    for fo in ex.frontier:
        if fo.homo_ev is None or fo.lumo_ev is None:
            continue
        tag = "" if fo.channel == "restricted" else f" ({fo.channel})"
        print(f"{'HOMO/LUMO' + tag:<20}: {fo.homo_ev:.3f} / {fo.lumo_ev:.3f} eV, "
              f"gap {fo.gap_ev:.3f} eV")
    if ex.mulliken:
        print(f"Mulliken charges    : {chg(ex.mulliken)}")
    if ex.loewdin:
        print(f"Loewdin charges     : {chg(ex.loewdin)}")
    if ex.opt_steps:
        jobs = len({s.job for s in ex.opt_steps})
        first, last = ex.opt_steps[0].energy_eh, ex.opt_steps[-1].energy_eh
        extra = f", {jobs} optimisations" if jobs > 1 else ""
        print(f"optimisation cycles : {len(ex.opt_steps)}{extra}")
        if first is not None and last is not None:
            print(f"                      E first {first:.6f}, last {last:.6f} Eh")
    if ex.ir_modes:
        strongest = max(ex.ir_modes, key=lambda m: m.intensity_km_mol)
        print(f"strongest IR band   : {strongest.freq_cm1:.1f} cm-1 "
              f"({strongest.intensity_km_mol:.1f} km/mol)")


def _emit(text: str, path: Path | None) -> None:
    if path:
        try:
            path.write_text(text)
        except OSError as exc:
            raise CliError(f"cannot write {path}: {exc}") from None
        print(f"wrote {path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _auto_resources(spec: CalcSpec) -> CalcSpec:
    res = suggest_resources(detect_hardware())
    print(f"[resources] nprocs={res.nprocs}, maxcore={res.maxcore_mb} MB per core (auto)",
          file=sys.stderr)
    return spec.model_copy(update={"resources": res})


def cmd_resources(args) -> int:
    hw = detect_hardware()
    sug = suggest_resources(hw)

    def gib(mb):
        return "n/a" if mb is None else f"{mb / 1024:.1f} GiB"

    print(f"cores (physical)    : {hw.cores}")
    print(f"RAM total           : {gib(hw.ram_mb)}")
    print(f"RAM available       : {gib(hw.available_mb)}")
    print(f"suggested           : nprocs={sug.nprocs}, maxcore={sug.maxcore_mb} MB per core")
    print(f"                      ({sug.nprocs} x {sug.maxcore_mb} MB = "
          f"{gib(sug.nprocs * sug.maxcore_mb)} in total)")
    return 0


def cmd_render(args) -> int:
    spec = load_spec(args.spec)
    if args.auto_resources:
        spec = _auto_resources(spec)
    _print_lint(spec)
    _emit(render_input(spec), args.output)
    return 0


def cmd_ask(args) -> int:
    from .llm import make_backend, plan_calculation

    text = _read_text(args.xyz, "XYZ file")
    try:
        atoms = atoms_from_xyz(text)
    except ValueError as exc:
        raise CliError(f"{args.xyz}: {exc}") from None
    backend = make_backend(args.backend, args.model, args.base_url, args.api_key_env)
    planned = plan_calculation(args.request, atoms, backend=backend)
    for note in planned.assumptions:
        print(f"[assumption] {note}", file=sys.stderr)
    spec = _auto_resources(planned.spec) if args.auto_resources else planned.spec
    _print_lint(spec)
    if args.spec_out:
        args.spec_out.write_text(spec.model_dump_json(indent=2))
        print(f"wrote {args.spec_out}", file=sys.stderr)
    _emit(render_input(spec), args.output)
    return 0


def cmd_run(args) -> int:
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 3
    orca = shutil.which("orca")
    if not orca:
        print("ORCA not found in PATH.", file=sys.stderr)
        return 2
    inp = args.input.resolve()
    out = inp.with_suffix(".out")
    print(f"running {orca} {inp.name} -> {out.name}", file=sys.stderr)
    with open(out, "w") as fh:
        subprocess.run([orca, inp.name], cwd=inp.parent, stdout=fh, stderr=subprocess.STDOUT)
    res = parse_file(out)
    print_summary(res)
    print_extras(extract(out.read_text(errors="replace")))
    return 0 if not res.problems else 1


def cmd_xyz(args) -> int:
    geoms = parse_geometries(_read_text(args.outfile, "ORCA output"))
    if not geoms:
        print("no geometry found in the file", file=sys.stderr)
        return 1
    if args.all:
        body = "".join(write_xyz(g, f"frame {i} of {len(geoms)}") for i, g in enumerate(geoms, 1))
    else:
        body = write_xyz(geoms[-1], f"{args.outfile.name}, last geometry")
    _emit(body, args.xyz_out)
    return 0


def cmd_table(args) -> int:
    entries = []
    for path in args.files:
        _read_text(path, "ORCA output")  # fail fast with one clear line
        entries.append((path.stem, parse_file(path)))
    try:
        table = build_rows(entries, args.temperature, args.by, args.sort)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for w in table.warnings:
        print(f"[warning] {w}", file=sys.stderr)
    _emit(render_table(table, args.format), args.output)
    return 0


def cmd_watch(args) -> int:
    try:
        return watch(args.file, interval=args.interval, once=args.once,
                     tty=False if args.plain else None)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 0


def cmd_parse(args) -> int:
    text = _read_text(args.output, "ORCA output")
    res = parse_file(args.output)
    extras = extract(text)
    if args.json:
        print(json.dumps({**res.to_dict(), **extras.to_dict()}, indent=2, ensure_ascii=False))
        return 0
    print_summary(res)
    print_extras(extras)
    return 0 if not res.problems else 1


def cmd_check(args) -> int:
    text = _read_text(args.infile, "ORCA input")
    hardware = None if args.no_hardware else detect_hardware()
    report = parse_inp(text, args.infile, hardware)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return report.exit_code()

    inp = report.inp
    print(f"file                : {args.infile}")
    if inp.simple_line is not None:
        print(f"simple input        : ! {inp.simple_line}  (line {inp.simple_line_no})")
    if inp.resources is not None:
        r = inp.resources
        print(f"resources           : nprocs={r.nprocs}, maxcore={r.maxcore_mb} MB per core "
              f"({r.nprocs * r.maxcore_mb / 1024:.1f} GiB in total)")
    else:
        print("resources           : none, ORCA uses a single process")
    if inp.atoms:
        print(f"molecule            : {len(inp.atoms)} atoms, charge {inp.charge}, "
              f"multiplicity {inp.multiplicity}")
    for scan in inp.scan:
        values = ", ".join(f"{v:g}" for v in scan.values)
        print(f"scan                : {scan.kind} {' '.join(map(str, scan.atoms))} = {values} "
              f"(line {scan.line_no})")
    for issue in report.issues:
        print(f"[{issue.level}] {issue.message}")
    print(f"result              : {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s)")
    return report.exit_code()


def cmd_batch(args) -> int:
    text = _read_text(args.input, "input file")
    if args.input.suffix.lower() == ".json":
        spec = load_spec(args.input)
        job_input = f"{args.input.stem}.inp"
        nprocs, maxcore = spec.resources.nprocs, spec.resources.maxcore_mb
    else:
        report = parse_inp(text, args.input)
        for issue in report.issues:
            if issue.level != "info":
                print(f"[{issue.level}] {issue.message}", file=sys.stderr)
        if report.errors:
            print(f"error: {args.input} has {len(report.errors)} error(s) and would not run; "
                  "no script written", file=sys.stderr)
            return 2
        job_input = args.input.name
        nprocs, maxcore = report.inp.nprocs, report.inp.maxcore_mb

    try:
        cfg = JobConfig(input_name=job_input, nprocs=nprocs or 1, maxcore_mb=maxcore,
                        job_name=args.job_name or "", time_limit=args.time_limit,
                        partition=args.partition, modules=tuple(args.module or ()),
                        nodes=args.nodes, scheduler=args.scheduler)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    script = make_script(cfg)
    for note in script.warnings:
        print(f"[warning] {note}", file=sys.stderr)
    _emit(script.text, args.output)
    return 0



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orca-assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="CalcSpec JSON -> .inp")
    r.add_argument("spec", type=Path)
    r.add_argument("-o", "--output", type=Path)
    r.add_argument("--auto-resources", action="store_true",
                   help="set nprocs/maxcore from this machine's cores and RAM")
    r.set_defaults(func=cmd_render)

    a = sub.add_parser("ask", help="plain-language request + .xyz -> .inp")
    a.add_argument("request", help='e.g. "optimise and compute frequencies, wB97X-D3, def2-TZVP, water"')
    a.add_argument("xyz", type=Path)
    a.add_argument("-o", "--output", type=Path)
    a.add_argument("--spec-out", type=Path, help="also save the validated spec as JSON")
    a.add_argument("--backend", choices=["ollama", "openai", "anthropic"], default="ollama",
                   help="ollama: local and free (default); openai: any OpenAI-compatible "
                        "server; anthropic: Claude API")
    a.add_argument("--model", help="model name (for ollama see `ollama list`)")
    a.add_argument("--base-url", help="server URL for --backend openai")
    a.add_argument("--api-key-env", help="env var holding the API key for --backend openai")
    a.add_argument("--auto-resources", action="store_true",
                   help="set nprocs/maxcore from this machine's cores and RAM")
    a.set_defaults(func=cmd_ask)

    rs = sub.add_parser("resources", help="show detected cores/RAM and suggested nprocs/maxcore")
    rs.set_defaults(func=cmd_resources)

    x = sub.add_parser("run", help="run ORCA on an .inp, then parse the result")
    x.add_argument("input", type=Path)
    x.set_defaults(func=cmd_run)

    ck = sub.add_parser("check", help="check a hand-written .inp: keywords, %% blocks, "
                                      "resources, coordinates, scan")
    ck.add_argument("infile", type=Path)
    ck.add_argument("--json", action="store_true", help="machine-readable report")
    ck.add_argument("--no-hardware", action="store_true",
                    help="skip the nprocs x maxcore check against this machine")
    ck.set_defaults(func=cmd_check)

    b = sub.add_parser("batch", help="SLURM or PBS script for an .inp (or a spec JSON)")
    b.add_argument("input", type=Path, help=".inp to submit, or a spec .json to render+run")
    b.add_argument("--scheduler", choices=["slurm", "pbs"], default="slurm")
    b.add_argument("--job-name", help="default: the input file name")
    b.add_argument("--partition", help="SLURM partition or PBS queue")
    b.add_argument("--time", dest="time_limit", default="24:00:00",
                   help="wall time, e.g. 24:00:00 or 2-12:00:00 (default 24:00:00)")
    b.add_argument("--nodes", type=int, default=1,
                   help="nodes; ORCA runs on one node by default (1)")
    b.add_argument("--module", action="append", metavar="MODULE",
                   help="module to load, repeatable (e.g. --module orca/6.1.1)")
    b.add_argument("-o", "--output", type=Path, help="write the script here instead of stdout")
    b.set_defaults(func=cmd_batch)

    xy = sub.add_parser("xyz", help="write the last (or all) geometries of an .out as .xyz")
    xy.add_argument("outfile", type=Path)
    xy.add_argument("-o", "--output", dest="xyz_out", type=Path)
    xy.add_argument("--all", action="store_true",
                    help="multi-frame .xyz of every geometry (optimisation trajectory)")
    xy.set_defaults(func=cmd_xyz)

    t = sub.add_parser("table", help="energy table from several .out files: dE/dG in kcal/mol, "
                                     "Boltzmann populations")
    t.add_argument("files", nargs="+", type=Path)
    t.add_argument("--format", choices=["md", "csv", "json"], default="md")
    t.add_argument("-o", "--output", type=Path)
    t.add_argument("--temperature", type=float, default=298.15, help="kelvin (default 298.15)")
    t.add_argument("--by", choices=["auto", "g", "e"], default="auto",
                   help="energy for relative values: g = Gibbs, e = electronic, "
                        "auto = Gibbs when every file has it")
    t.add_argument("--sort", action="store_true", help="most populated first")
    t.set_defaults(func=cmd_table)

    w = sub.add_parser("watch", help="live status of a running ORCA job from its .out file")
    w.add_argument("file", type=Path)
    w.add_argument("--interval", type=float, default=2.0, help="seconds between checks")
    w.add_argument("--once", action="store_true", help="print the status once and exit")
    w.add_argument("--plain", action="store_true",
                   help="no screen clearing: print a new block only when the status changes "
                        "(easy to copy)")
    w.set_defaults(func=cmd_watch)

    pa = sub.add_parser("parse", help="summarise an ORCA .out file")
    pa.add_argument("output", type=Path)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_parse)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # `orca-assistant parse job.out | head` closes the pipe early. Point the
        # descriptor at /dev/null so the flush at exit is silent, and close
        # nothing: stdout belongs to the caller.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


def parse_main(argv: list[str] | None = None) -> int:
    """Entry point kept for the `orca-parse` command."""
    return main(["parse", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
