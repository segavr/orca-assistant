"""Plots from ORCA output data: optimisation energy profile and IR spectrum.

Data preparation is plain Python. matplotlib (optional extra `plot`) is only
imported when a figure is actually drawn.
"""
from __future__ import annotations

from pathlib import Path

from .extract import IRMode, OptStep
from .table import HARTREE_TO_KCAL


class PlotUnavailable(RuntimeError):
    """matplotlib is not installed."""


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotUnavailable(
            "matplotlib is not installed; run: pip install matplotlib "
            '(or pip install "orca-assistant[plot]")') from exc
    return plt


def opt_profiles(steps: list[OptStep]) -> list[tuple[int, list[int], list[float]]]:
    """Per optimisation job: (job, cycles, E - E_last in kcal/mol)."""
    jobs: dict[int, list[tuple[int, float]]] = {}
    for s in steps:
        if s.energy_eh is not None:
            jobs.setdefault(s.job, []).append((s.cycle, s.energy_eh))
    out = []
    for job, pts in sorted(jobs.items()):
        last = pts[-1][1]
        out.append((job, [c for c, _ in pts],
                    [(e - last) * HARTREE_TO_KCAL for _, e in pts]))
    return out


def ir_curve(modes: list[IRMode], scale: float = 1.0, fwhm: float = 20.0,
             xmin: float = 400.0, xmax: float = 4000.0,
             step: float = 2.0) -> tuple[list[float], list[float]]:
    """Sum of Lorentzians (peak height = band intensity, km/mol)."""
    if fwhm <= 0 or step <= 0 or xmax <= xmin or scale <= 0:
        raise ValueError("fwhm, step and scale must be positive and xmax > xmin")
    peaks = [(m.freq_cm1 * scale, m.intensity_km_mol) for m in modes if m.freq_cm1 > 0]
    n = int(round((xmax - xmin) / step)) + 1
    xs = [xmin + i * step for i in range(n)]
    g2 = (fwhm / 2) ** 2
    ys = [sum(h * g2 / ((x - x0) ** 2 + g2) for x0, h in peaks) for x in xs]
    return xs, ys


def plot_opt(steps: list[OptStep], path: Path) -> Path:
    profiles = [p for p in opt_profiles(steps) if len(p[1]) >= 2]
    if not profiles:
        raise ValueError("need at least 2 optimisation cycles with energies to plot")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    for job, cycles, rel in profiles:
        label = f"optimisation {job}" if len(profiles) > 1 else None
        ax.plot(cycles, rel, marker="o", label=label)
    ax.set_xlabel("optimisation cycle")
    ax.set_ylabel("E - E(last cycle), kcal/mol")
    ax.set_title("Geometry optimisation")
    ax.grid(alpha=0.3)
    if len(profiles) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ir(modes: list[IRMode], path: Path, scale: float = 1.0,
            fwhm: float = 20.0) -> Path:
    real = [m for m in modes if m.freq_cm1 > 0]
    if not real:
        raise ValueError("no IR data in the output (the calculation needs Freq)")
    xs, ys = ir_curve(real, scale=scale, fwhm=fwhm)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, linewidth=1.2)
    ax.vlines([m.freq_cm1 * scale for m in real], 0,
              [m.intensity_km_mol for m in real], alpha=0.5, linewidth=1)
    ax.set_xlim(max(xs), min(xs))  # IR convention: high wavenumbers on the left
    ax.set_ylim(bottom=0)
    ax.set_xlabel("wavenumber, cm$^{-1}$" + (f" (scaled x {scale:g})" if scale != 1 else ""))
    ax.set_ylabel("intensity, km/mol")
    ax.set_title("Calculated IR spectrum")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
