from pathlib import Path

import pytest

from orca_assistant.jobqueue import (
    JobResult, all_ok, format_results, read_inp_resources, run_queue,
)
from orca_assistant.resources import Hardware

HW = Hardware(cores=4, ram_mb=64000, available_mb=60000)


class FakePopen:
    live: list = []
    peak = 0
    started: list = []

    def __init__(self, args, cwd=None, stdout=None, stderr=None):
        self.fail = "FAIL" in (Path(cwd) / args[1]).read_text()
        self.polls = 0
        self.returncode = None
        self.terminated = False
        FakePopen.live.append(self)
        FakePopen.started.append(args[1])
        FakePopen.peak = max(FakePopen.peak, len(FakePopen.live))
        if self.fail:
            stdout.write("[file orca_main/run.cpp, line 1]: boom\n")
        else:
            stdout.write("FINAL SINGLE POINT ENERGY   -1.500000000000\n"
                         "ORCA TERMINATED NORMALLY\n")
        stdout.flush()

    def _end(self):
        if self in FakePopen.live:
            FakePopen.live.remove(self)

    def poll(self):
        if self.terminated:
            return self.returncode
        self.polls += 1
        if self.polls < 3:
            return None
        self._end()
        self.returncode = 1 if self.fail else 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self._end()

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture(autouse=True)
def reset_fake():
    FakePopen.live, FakePopen.peak, FakePopen.started = [], 0, []


def make_inp(tmp_path, name, nprocs=1, maxcore=1000, extra=""):
    p = tmp_path / name
    p.write_text(f"! B3LYP\n%pal nprocs {nprocs} end\n%maxcore {maxcore}\n{extra}\n"
                 "* xyz 0 1\nH 0 0 0\n*\n")
    return p


def go(inputs, hw=HW, **kw):
    return run_queue(inputs, "/fake/orca", hw, popen=FakePopen,
                     sleep=lambda _: None, log=lambda m: None, **kw)


def test_read_inp_resources():
    assert read_inp_resources("%pal nprocs 4 end\n%maxcore 2000\n") == (4, 2000)
    assert read_inp_resources("! B3LYP PAL8\n* xyz 0 1\n") == (8, 1024)
    assert read_inp_resources("! B3LYP\n") == (1, 1024)
    assert read_inp_resources("%PAL\n  NPROCS 2\nEND\n%MAXCORE 500\n") == (2, 500)


def test_default_is_sequential(tmp_path):
    a, b = make_inp(tmp_path, "a.inp"), make_inp(tmp_path, "b.inp")
    results, interrupted = go([a, b])
    assert [r.status for r in results] == ["done", "done"]
    assert FakePopen.peak == 1 and not interrupted
    assert results[0].energy_eh == pytest.approx(-1.5)


def test_parallel_when_it_fits(tmp_path):
    a = make_inp(tmp_path, "a.inp", nprocs=2)
    b = make_inp(tmp_path, "b.inp", nprocs=2)
    go([a, b], max_jobs=2)
    assert FakePopen.peak == 2


def test_cores_limit_parallelism(tmp_path):
    a = make_inp(tmp_path, "a.inp", nprocs=3)
    b = make_inp(tmp_path, "b.inp", nprocs=3)
    results, _ = go([a, b], max_jobs=2)
    assert FakePopen.peak == 1 and all_ok(results)


def test_memory_limit_parallelism(tmp_path):
    hw = Hardware(cores=8, ram_mb=64000, available_mb=20000)  # budget 15000 MB
    small = [make_inp(tmp_path, f"s{i}.inp", maxcore=5000) for i in range(2)]
    go(small, hw=hw, max_jobs=2)
    assert FakePopen.peak == 2
    FakePopen.live, FakePopen.peak = [], 0
    big = [make_inp(tmp_path, f"g{i}.inp", maxcore=10000) for i in range(2)]
    go(big, hw=hw, max_jobs=2)
    assert FakePopen.peak == 1


def test_skips_finished_unless_forced(tmp_path):
    a = make_inp(tmp_path, "a.inp")
    a.with_suffix(".out").write_text("ORCA TERMINATED NORMALLY\n")
    results, _ = go([a])
    assert results[0].status == "skipped" and FakePopen.started == []
    results, _ = go([a], force=True)
    assert results[0].status == "done" and FakePopen.started == ["a.inp"]


def test_missing_input_is_error(tmp_path):
    results, _ = go([tmp_path / "none.inp"])
    assert results[0].status == "error" and not all_ok(results)


def test_failed_job_does_not_stop_the_queue(tmp_path):
    a = make_inp(tmp_path, "a.inp", extra="FAIL")
    b = make_inp(tmp_path, "b.inp")
    results, _ = go([a, b])
    assert [r.status for r in results] == ["failed", "done"]
    assert FakePopen.started == ["a.inp", "b.inp"] and not all_ok(results)


def test_interrupt_terminates_running_and_marks_rest(tmp_path):
    a, b = make_inp(tmp_path, "a.inp"), make_inp(tmp_path, "b.inp")

    def boom(_):
        raise KeyboardInterrupt

    results, interrupted = run_queue([a, b], "/fake/orca", HW, popen=FakePopen,
                                     sleep=boom, log=lambda m: None)
    assert interrupted and FakePopen.live == []
    assert [r.status for r in results] == ["interrupted", "not started"]


def test_job_larger_than_budget_runs_alone_with_warning(tmp_path):
    hw = Hardware(cores=2, ram_mb=64000, available_mb=60000)
    a = make_inp(tmp_path, "a.inp", nprocs=4)
    results, _ = go([a], hw=hw)
    assert results[0].status == "done"
    assert any("budget" in n for n in results[0].notes)


def test_format_results():
    text = format_results([JobResult("a", "done", 12.4, -1.5),
                           JobResult("b", "failed", None, None, ["exit code 1"])])
    lines = text.splitlines()
    assert lines[0].startswith("job") and "done" in lines[1] and "-1.50000000" in lines[1]
    assert "failed" in lines[2] and "exit code 1" in lines[2]
