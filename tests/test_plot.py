import sys
from pathlib import Path

import pytest

from orca_assistant.extract import IRMode, OptStep, extract
from orca_assistant.plot import PlotUnavailable, ir_curve, opt_profiles, plot_ir, plot_opt
from orca_assistant.table import HARTREE_TO_KCAL

WATER = Path(__file__).parent / "data" / "water.out"
needs_water = pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")


def test_opt_profiles_relative_to_last_cycle():
    steps = [OptStep(1, 1, -1.000), OptStep(1, 2, -1.001), OptStep(1, 3, -1.002)]
    (job, cycles, rel), = opt_profiles(steps)
    assert (job, cycles) == (1, [1, 2, 3])
    assert rel[-1] == 0 and abs(rel[0] - 0.002 * HARTREE_TO_KCAL) < 1e-9


def test_opt_profiles_split_jobs_and_skip_missing_energy():
    steps = [OptStep(1, 1, -1.0), OptStep(1, 2, None), OptStep(2, 1, -2.0), OptStep(2, 2, -2.1)]
    profiles = opt_profiles(steps)
    assert [p[0] for p in profiles] == [1, 2]
    assert profiles[0][1] == [1] and profiles[1][1] == [1, 2]


def test_ir_curve_peak_height_and_width():
    xs, ys = ir_curve([IRMode(1, 1000.0, 100.0)], fwhm=20.0)
    at = dict(zip(xs, ys))
    assert at[1000.0] == pytest.approx(100.0)
    assert at[1010.0] == pytest.approx(50.0) and at[990.0] == pytest.approx(50.0)


def test_ir_curve_scale_and_negative_modes():
    modes = [IRMode(1, 1000.0, 100.0), IRMode(2, -300.0, 500.0)]
    xs, ys = ir_curve(modes, scale=0.96, fwhm=20.0)
    assert xs[ys.index(max(ys))] == pytest.approx(960.0, abs=2.0)
    assert max(ys) == pytest.approx(100.0, rel=0.02)


def test_ir_curve_bad_arguments():
    for kw in ({"fwhm": 0}, {"scale": -1}, {"step": 0}, {"xmin": 500, "xmax": 400}):
        with pytest.raises(ValueError):
            ir_curve([IRMode(1, 1000.0, 1.0)], **kw)


def test_plot_functions_report_data_problems(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        plot_opt([OptStep(1, 1, -1.0)], tmp_path / "x.png")
    with pytest.raises(ValueError, match="no IR data"):
        plot_ir([], tmp_path / "x.png")


def test_missing_matplotlib_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(PlotUnavailable, match="pip install"):
        plot_ir([IRMode(1, 1000.0, 10.0)], tmp_path / "x.png")


PNG = b"\x89PNG\r\n\x1a\n"


@needs_water
def test_water_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    ex = extract(WATER.read_text())
    a = plot_opt(ex.opt_steps, tmp_path / "opt.png")
    b = plot_ir(ex.ir_modes, tmp_path / "ir.png", scale=0.97)
    assert a.read_bytes().startswith(PNG) and b.read_bytes().startswith(PNG)
