from pathlib import Path

import pytest

from orca_assistant.cli import main

WATER = Path(__file__).parent / "data" / "water.out"
PNG = b"\x89PNG\r\n\x1a\n"


@pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")
@pytest.mark.parametrize("kind", ["opt", "ir"])
def test_plot_writes_png(kind, tmp_path):
    pytest.importorskip("matplotlib")
    dest = tmp_path / f"{kind}.png"
    assert main(["plot", kind, str(WATER), "-o", str(dest)]) == 0
    assert dest.read_bytes().startswith(PNG)


def test_plot_ir_without_frequencies_is_an_error(tmp_path, capsys):
    f = tmp_path / "job.out"
    f.write_text("ORCA TERMINATED NORMALLY\n")
    assert main(["plot", "ir", str(f)]) == 1
    assert "no IR data" in capsys.readouterr().err
