# orca-assistant

[![tests](https://github.com/segavr/orca-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/segavr/orca-assistant/actions/workflows/tests.yml)

[![tests](https://github.com/segavr/orca-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/segavr/orca-assistant/actions/workflows/tests.yml)

Typed spec -> validated ORCA input. The LLM layer (planned) only produces a
`CalcSpec`; all ORCA syntax comes from deterministic code.

```
natural language --(LLM)--> CalcSpec --(validate)--> render_input --> .inp
                                          |
                                        lint()
```

## Quick start
```bash
pip install -e ".[dev]"
pytest

orca-assistant render examples/water_opt_freq.json -o water.inp   # spec -> .inp
orca-assistant run water.inp                                       # run ORCA + summary
orca-assistant parse water.out [--json]                            # summarise any .out
```

## Plain-language requests (LLM layer)
The model never sees or writes coordinates: it fills a structured request
(task, method, resources, charge, multiplicity). The result is validated, merged
with the atoms from your XYZ file, and validation errors go back to the model for
a retry. Every default it chose is printed as an `[assumption]`; `--spec-out`
stores the exact spec for reproducibility.

**Free and local (default): Ollama**
```bash
ollama pull <model>          # a 3-8B instruct model; see `ollama list`
orca-assistant ask "optimise and compute frequencies, wB97X-D3, def2-TZVP, water" mol.xyz \
    --model <model> -o job.inp --spec-out job.json
```
**Any OpenAI-compatible server** (llama.cpp, LM Studio, a provider's free tier):
```bash
export GROQ_API_KEY=...
orca-assistant ask "..." mol.xyz --backend openai \
    --base-url https://api.groq.com/openai/v1 --model <model> --api-key-env GROQ_API_KEY
```
**Claude API** (paid): `pip install -e ".[llm]"`, `export ANTHROPIC_API_KEY=...`,
then `--backend anthropic`.

Small local models can return valid JSON with wrong content, so the validator and
retry loop matter; the benchmark (roadmap) measures how often each model is right.

## Roadmap
1. [x] Spec + renderer + linter (this MVP)
2. [x] Output parser: `orca-parse file.out` (convergence, imaginary freqs, energy, thermochemistry, warnings)
3. [x] `run` command: execute ORCA, parse, report
4. [x] LLM layer: text -> CalcSpec, pluggable backends (Ollama, OpenAI-compatible, Claude)
4b. [ ] RAG over the ORCA manual, benchmark of request -> input pairs
4c. [ ] SLURM script generator
5. [ ] Error diagnosis and guarded auto-restart
6. [ ] Benchmark set of reference inputs
7. [ ] MCP server exposing generate / lint / parse tools

## Resources

`orca-assistant resources` prints the detected physical cores and RAM and a
safe `nprocs` / `maxcore` suggestion (`%maxcore` is per core, so the total is
`nprocs x maxcore`). `render` and `ask` accept `--auto-resources` to apply it,
and the linter warns when the requested `nprocs x maxcore` does not fit the
machine. Detection uses only the standard library (Linux, macOS, Windows).

## Reading outputs

`orca-assistant parse file.out` now also reports the ORCA version, HOMO/LUMO
and gap (per spin channel for open-shell systems), Mulliken and Loewdin
charges, optimisation cycles and the strongest IR band; `--json` includes all
of it, with the per-cycle energies and IR intensities. `orca-assistant xyz
file.out` writes the last geometry as `.xyz` (`--all` writes the whole
optimisation trajectory as a multi-frame file).

## Energy tables

`orca-assistant table a.out b.out ... [--format md|csv|json] [--temperature K]
[--by auto|g|e] [--sort] [-o file]` collects energies from several outputs:
E, ZPE, H and G in Eh, relative energies in kcal/mol and Boltzmann
populations. Relative values use the Gibbs free energy when every file has it,
otherwise the electronic energy (with a warning). Only compare calculations
done at the same level of theory.

## Watching a running job

`orca-assistant watch job.out` follows the output of a running ORCA job:
current stage (SCF, geometry optimisation, final single point), optimisation
cycle, last energy, the geometry convergence table and the time per cycle. It
exits when ORCA finishes (exit code 0) or fails (1); `--once` prints a single
snapshot and `--plain` prints a new block only when the status changes (no
screen clearing, easy to copy). No remaining-time estimate is given because
the number of optimisation cycles is not known in advance. Tested live on a
benzene optimisation + frequencies run with ORCA 6.1.1.

## Plots

`orca-assistant plot opt file.out` draws the energy of each optimisation cycle
relative to the last one (kcal/mol); `orca-assistant plot ir file.out
[--scale 0.97] [--fwhm 20]` draws the calculated IR spectrum with Lorentzian
broadening. Plots need matplotlib: `pip install "orca-assistant[plot]"`. The
frequency scale factor depends on the functional and basis set; the default is
1.0 (unscaled).

## Job queue

`orca-assistant queue a.inp b.inp ... [--jobs N] [--max-procs N]
[--max-mem-mb MB] [--force]` runs several inputs. By default they run one at a
time; with `--jobs N` up to N run together, but only while the total `nprocs`
and `nprocs x maxcore` (read from each input) fit the detected cores and the
available memory. A job larger than the budget runs alone, with a warning.
Inputs whose `.out` already finished normally are skipped, so rerunning the
same command after a crash resumes the queue (`--force` recomputes
everything). A failed job does not stop the queue; a summary table is printed
at the end and `Ctrl+C` stops the running jobs cleanly. Sequential runs,
skipping and failure handling were tested live with ORCA 6.1.1; the parallel
mode is covered by tests with a fake process only.

## Cluster scripts (SLURM / PBS)

`orca-assistant submit job.inp --scheduler slurm|pbs [--time HH:MM:SS]
[--partition P] [--account A] [--module M ...] [--orca /full/path/orca]
[--scratch /scratch/$USER] [--copy-files F ...] [--submit]` writes a batch
script next to the input. Cores and `%maxcore` are read from the input; the
memory request is `nprocs x maxcore x 1.3` (ORCA's `%maxcore` is a soft limit)
unless `--mem-mb` is given. The script calls ORCA by its full path (required
for parallel runs); with `--scratch` it runs in `<DIR>/<jobid>`, copies the
results back (also when ORCA fails, keeping ORCA's exit status) and removes
the scratch directory.

Status: the generated scripts are tested with `bash -n`, with a fake ORCA and
with ORCA 6.1.1 run locally through the script, but they have not been
submitted to a real SLURM or PBS scheduler. Module names, partitions and
whether `module` works in non-login shells differ between clusters, so check
the script once before relying on it. PBS support targets PBS Pro / OpenPBS
(`select=...` syntax); Torque is not supported.

## Troubleshooting

**`orca` is not ORCA.** On many desktop Linux systems `/usr/bin/orca` is the
GNOME screen reader. `run`, `queue` and the generated batch scripts use the
first `orca` found in `PATH`: check with `which -a orca` that the
quantum-chemistry ORCA comes first, or pass its full path
(`submit --orca /path/to/orca`).

## Diagnosing failed runs

`orca-assistant diagnose file.out [--json]` explains what went wrong or looks
suspicious: unrecognised keywords (e.g. a misspelt basis set), impossible
charge/multiplicity (with the valid multiplicities), SCF that did not
converge, optimisation stopped at the cycle limit, imaginary frequencies in a
minimum search (or a wrong number in a transition-state search), and output
that ends without a termination message. Each finding has the evidence from
the output and a suggested remedy. The rules read the text of the output, not
the exit code: ORCA can stop with an error and still return 0. The message
formats come from real ORCA 6.1.1 outputs (kept in `tests/data/fail_*.out`);
other ORCA versions may word them differently. Running out of memory is only
reported as a guess ("output ends abruptly"): ORCA accepted even `%maxcore 10`
in our tests and we have no real out-of-memory output yet.

## Fixing a failed run

`orca-assistant fix job.out [--inp job.inp] [--multiplicity N] [--geometry
last.xyz] [-o new.inp]` writes a corrected copy of the input next to the
original (`job_fix1.inp`, then `job_fix2.inp`, ...; nothing is ever
overwritten) and prints the exact diff. Only changes that do not alter the
physics are made automatically: `SlowConv` (then `VerySlowConv`) and a higher
`%scf MaxIter` for a non-converged SCF; the last geometry and a higher
`%geom MaxIter` for an optimisation that hit the cycle limit. A new
multiplicity is applied only when you ask for it with `--multiplicity` after
a charge/multiplicity error, because it sets the spin state. Unknown keywords
get "did you mean" suggestions from a list of common names but are not
replaced. Imaginary frequencies are not fixed automatically yet.

## Careful automatic restart

`orca-assistant auto job.inp [--max-attempts 3]` runs ORCA, diagnoses the
output and, if the problem has a safe automatic fix (SCF convergence aids, a
restart after the optimisation cycle limit), writes `job_fix1.inp` and runs
again, up to `--max-attempts` runs in total (default 3, at most 10). It stops
immediately when something needs your decision (multiplicity or charge,
unknown keywords, imaginary frequencies, errors without a rule) or when
nothing more can be fixed automatically. Nothing is overwritten: every
attempt has its own input and output, and a journal `job_auto1.md` records
the findings, the changes with their diff and the outcome of each attempt.
Exit code 0 means the last attempt ran without problems.

## Checking a hand-written input

`orca-assistant check job.inp [--json]` checks an input before you run it:
impossible charge/multiplicity (computed from the atoms, exactly the error ORCA
raises), a missing `!` line or coordinates, an unclosed coordinate block or
`%` block, a missing `* xyzfile`, duplicated keywords, `-3c` methods given a
basis set, double-counted dispersion, very low `MaxIter` limits, `OptTS`
without a Hessian, and `nprocs x maxcore` against the cores and RAM of the
machine. Possible misspellings are only hints, because the list of common
keyword names is not exhaustive. Exit code 1 means at least one error.
Differences between ORCA versions are not checked yet.

## Which ORCA is used

`run`, `queue` and `auto` use the first program called `orca` in `PATH` that is
not the GNOME screen reader (on many desktop Linux systems `/usr/bin/orca` is a
Python script that imports the `orca` package; it is recognised and skipped, so
it is never started by mistake). A program with ORCA's helper programs
(`orca_*`) next to it is reported as confirmed; one without them is still used,
with a short warning. To choose explicitly, set
`ORCA_ASSISTANT_ORCA=/full/path/orca` (it is used without any check).
`orca-assistant doctor` shows the choice, the skipped candidates, optional
packages, cores, memory and the cluster tools found on the machine.
