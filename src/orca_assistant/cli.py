from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .batch import DEFAULT_TIME, SCHEDULERS, SUBMIT_COMMAND, SUFFIX, BatchSpec, build_script
from .batch import submit as submit_script
from .diagnose import diagnose, format_findings
from .extract import extract, parse_geometries, write_xyz
from .lint import lint
from .parser import ParsedOutput, parse_file
from .jobqueue import all_ok, format_results, run_queue
from .plot import PlotUnavailable, plot_ir, plot_opt
from .render import render_input
from .resources import detect_hardware, suggest_resources
from .spec import CalcSpec, atoms_from_xyz
from .table import build_rows, render_table
from .watch import watch


def _print_lint(spec: CalcSpec) -> None:
    for issue in lint(spec, detect_hardware()):
        print(f"[{issue.level}] {issue.message}", file=sys.stderr)


def print_summary(res: ParsedOutput) -> None:
    def fmt(v, unit=""):
        return "n/a" if v is None else f"{v:.6f}{unit}"

    print(f"terminated normally : {res.terminated_normally}")
    print(f"optimisation        : {res.opt_converged}")
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
        path.write_text(text)
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
    spec = CalcSpec.model_validate_json(args.spec.read_text())
    if args.auto_resources:
        spec = _auto_resources(spec)
    _print_lint(spec)
    _emit(render_input(spec), args.output)
    return 0


def cmd_ask(args) -> int:
    from .llm import make_backend, plan_calculation

    atoms = atoms_from_xyz(args.xyz.read_text())
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
    geoms = parse_geometries(args.outfile.read_text(errors="replace"))
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
    entries = [(p.stem, parse_file(p)) for p in args.files]
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


def cmd_plot(args) -> int:
    ex = extract(args.file.read_text(errors="replace"))
    dest = args.output or args.file.with_name(f"{args.file.stem}_{args.kind}.png")
    try:
        if args.kind == "opt":
            plot_opt(ex.opt_steps, dest)
        else:
            plot_ir(ex.ir_modes, dest, scale=args.scale, fwhm=args.fwhm)
    except (PlotUnavailable, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {dest}", file=sys.stderr)
    return 0


def cmd_queue(args) -> int:
    orca = shutil.which("orca")
    if not orca:
        print("ORCA not found in PATH.", file=sys.stderr)
        return 2
    results, interrupted = run_queue(args.inputs, orca, detect_hardware(),
                                     max_jobs=args.jobs, max_procs=args.max_procs,
                                     max_mem_mb=args.max_mem_mb, force=args.force)
    sys.stdout.write(format_results(results))
    if interrupted:
        return 130
    return 0 if all_ok(results) else 1


def cmd_submit(args) -> int:
    if not args.input.exists():
        print(f"input file not found: {args.input}", file=sys.stderr)
        return 3
    spec = BatchSpec(args.scheduler, args.input, time=args.time, mem_mb=args.mem_mb,
                     partition=args.partition, account=args.account,
                     modules=args.module or [], orca=args.orca, scratch=args.scratch,
                     copy_files=args.copy_files or [], name=args.name)
    try:
        text, notes = build_script(spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    dest = args.output or args.input.with_suffix(SUFFIX[args.scheduler])
    dest.write_text(text)
    for note in notes:
        print(f"[note] {note}", file=sys.stderr)
    print(f"wrote {dest}", file=sys.stderr)
    if args.submit:
        try:
            code, msg = submit_script(dest, args.scheduler)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(msg)
        return code
    print(f"submit with: {SUBMIT_COMMAND[args.scheduler]} {dest}", file=sys.stderr)
    return 0


def cmd_diagnose(args) -> int:
    if not args.file.exists():
        print(f"file not found: {args.file}", file=sys.stderr)
        return 3
    findings = diagnose(args.file.read_text(errors="replace"))
    if args.json:
        print(json.dumps([f.to_dict() for f in findings], indent=2, ensure_ascii=False))
    else:
        sys.stdout.write(format_findings(findings))
    return 1 if findings else 0


def cmd_parse(args) -> int:
    res = parse_file(args.output)
    extras = extract(args.output.read_text(errors="replace"))
    if args.json:
        print(json.dumps({**res.to_dict(), **extras.to_dict()}, indent=2, ensure_ascii=False))
        return 0
    print_summary(res)
    print_extras(extras)
    return 0 if not res.problems else 1


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

    pl = sub.add_parser("plot", help="PNG plot from an .out file: optimisation profile or IR spectrum")
    pl.add_argument("kind", choices=["opt", "ir"])
    pl.add_argument("file", type=Path)
    pl.add_argument("-o", "--output", type=Path, help="default: <file>_<kind>.png next to the .out")
    pl.add_argument("--scale", type=float, default=1.0,
                    help="frequency scaling factor for the IR plot (default 1.0, i.e. unscaled)")
    pl.add_argument("--fwhm", type=float, default=20.0,
                    help="IR band width in cm-1 (default 20)")
    pl.set_defaults(func=cmd_plot)

    q = sub.add_parser("queue", help="run several .inp files one after another "
                                     "(or in parallel if cores and memory allow)")
    q.add_argument("inputs", nargs="+", type=Path)
    q.add_argument("--jobs", type=int, default=1,
                   help="max simultaneous jobs (default 1); limited by cores and memory")
    q.add_argument("--max-procs", type=int, help="core budget (default: detected physical cores)")
    q.add_argument("--max-mem-mb", type=int,
                   help="memory budget in MB (default: 75%% of the RAM available now)")
    q.add_argument("--force", action="store_true",
                   help="rerun jobs whose .out already finished normally")
    q.set_defaults(func=cmd_queue)

    sb = sub.add_parser("submit", help="write a SLURM or PBS batch script for an .inp "
                                       "(cores and memory are read from the input)")
    sb.add_argument("input", type=Path)
    sb.add_argument("--scheduler", required=True, choices=list(SCHEDULERS),
                    help="slurm, or pbs (PBS Pro / OpenPBS; Torque is not supported)")
    sb.add_argument("--time", default=DEFAULT_TIME,
                    help=f"wall time, HH:MM:SS or D-HH:MM:SS (default {DEFAULT_TIME})")
    sb.add_argument("--mem-mb", type=int,
                    help="total memory to request (default: nprocs x maxcore x 1.3)")
    sb.add_argument("--partition", "--queue", dest="partition",
                    help="SLURM partition / PBS queue")
    sb.add_argument("--account", help="accounting project")
    sb.add_argument("--module", action="append",
                    help="module to load (repeat for several, e.g. --module openmpi/4.1.8)")
    sb.add_argument("--orca", help="absolute path to the orca executable "
                                   "(default: `orca` found in PATH when the job runs)")
    sb.add_argument("--scratch", help="run in <DIR>/<jobid> and copy results back, "
                                      "e.g. /scratch/$USER")
    sb.add_argument("--copy-files", action="append",
                    help="extra input file to copy to scratch (repeatable; needs --scratch)")
    sb.add_argument("--name", help="job name (default: the input's name)")
    sb.add_argument("-o", "--output", type=Path, help="script path (default: next to the .inp)")
    sb.add_argument("--submit", action="store_true",
                    help="also run sbatch/qsub (only on a cluster login node)")
    sb.set_defaults(func=cmd_submit)

    dg = sub.add_parser("diagnose", help="explain what went wrong (or looks suspicious) "
                                         "in an ORCA .out file")
    dg.add_argument("file", type=Path)
    dg.add_argument("--json", action="store_true")
    dg.set_defaults(func=cmd_diagnose)

    pa = sub.add_parser("parse", help="summarise an ORCA .out file")
    pa.add_argument("output", type=Path)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_parse)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def parse_main(argv: list[str] | None = None) -> int:
    """Entry point kept for the `orca-parse` command."""
    return main(["parse", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
