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
