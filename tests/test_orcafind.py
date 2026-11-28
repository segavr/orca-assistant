import os
import sys

import pytest

from orca_assistant.doctor import run_doctor
from orca_assistant.orcafind import ENV_OVERRIDE, classify, find_orca
from orca_assistant.resources import Hardware

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permissions")


def make(dirpath, name, content=b"\x7fELF binary", mode=0o755):
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_bytes(content)
    p.chmod(mode)
    return p


def real_install(root):
    make(root, "orca")
    make(root, "orca_scf")
    make(root, "orca_leanscf")
    return root / "orca"


def screen_reader(root):
    return make(root, "orca", content=b"#!/usr/bin/python3\n#\n# Orca\n#\n"
                b"# Copyright 2010-2012 The Orca Team\nfrom orca import debug\n")


def env(*dirs):
    return {"PATH": os.pathsep.join(str(d) for d in dirs)}


def test_classify(tmp_path):
    assert classify(real_install(tmp_path / "qc"))[0] == "real"
    assert classify(screen_reader(tmp_path / "sys"))[0] == "screen_reader"
    assert classify(make(tmp_path / "scripts", "orca", content=b"#!/bin/sh\nexec x\n"))[0] == "unconfirmed"
    assert classify(make(tmp_path / "lone", "orca"))[0] == "unconfirmed"


def test_real_orca_wins_over_the_screen_reader_even_when_later_in_path(tmp_path):
    sr = screen_reader(tmp_path / "usr_bin")
    real = real_install(tmp_path / "qc")
    res = find_orca(env(tmp_path / "usr_bin", tmp_path / "qc"))
    assert res.path == str(real) and res.verdict == "real"
    assert len(res.skipped) == 1 and str(sr) in res.skipped[0] and "screen reader" in res.skipped[0]


def test_only_the_screen_reader_means_not_found_with_an_explanation(tmp_path):
    screen_reader(tmp_path / "usr_bin")
    res = find_orca(env(tmp_path / "usr_bin"))
    assert res.path is None and res.verdict == "none" and "screen reader" in res.skipped[0]


def test_a_lone_binary_is_used_but_marked_unconfirmed(tmp_path):
    lone = make(tmp_path / "lone", "orca")
    res = find_orca(env(tmp_path / "lone"))
    assert res.path == str(lone) and res.verdict == "unconfirmed"


def test_wrapper_script_next_to_helpers_counts_as_real(tmp_path):
    root = tmp_path / "site"
    make(root, "orca", content=b"#!/bin/bash\nexec real_orca \"$@\"\n")
    make(root, "orca_gtoint")
    assert find_orca(env(root)).verdict == "real"


def test_two_names_for_the_same_file_are_one_candidate(tmp_path):
    root = real_install(tmp_path / "qc").parent
    link_dir = tmp_path / "bin"
    link_dir.mkdir()
    (link_dir / "orca").symlink_to(root / "orca")
    res = find_orca(env(link_dir, root))
    assert res.verdict == "real" and res.skipped == []


def test_a_second_real_orca_is_reported_as_unused(tmp_path):
    first, second = real_install(tmp_path / "a"), real_install(tmp_path / "b")
    res = find_orca(env(tmp_path / "a", tmp_path / "b"))
    assert res.path == str(first) and "another ORCA" in res.skipped[0]


def test_non_executable_and_missing_entries_are_ignored(tmp_path):
    make(tmp_path / "x", "orca", mode=0o644)
    res = find_orca({"PATH": f"{tmp_path / 'x'}{os.pathsep}{tmp_path / 'nowhere'}{os.pathsep}"})
    assert res.path is None and res.skipped == []


def test_override_variable(tmp_path):
    lone = make(tmp_path / "mine", "orca")
    res = find_orca({**env(), ENV_OVERRIDE: str(lone)})
    assert res.path == str(lone) and res.verdict == "user"
    bad = find_orca({**env(), ENV_OVERRIDE: str(tmp_path / "missing")})
    assert bad.path is None and ENV_OVERRIDE in bad.problem


def test_override_beats_the_search(tmp_path):
    real_install(tmp_path / "qc")
    mine = make(tmp_path / "mine", "orca")
    res = find_orca({**env(tmp_path / "qc"), ENV_OVERRIDE: str(mine)})
    assert res.path == str(mine)


HW = Hardware(cores=4, ram_mb=6829, available_mb=3100)


def doctor(search, versions=None, tools=None):
    versions = versions if versions is not None else {"matplotlib": "3.11.2"}
    tools = tools if tools is not None else {"bash": "/usr/bin/bash"}
    return run_doctor(lambda: search, lambda: HW, lambda t: tools.get(t),
                      lambda m: versions.get(m), "3.12.3")


def test_doctor_report_with_confirmed_orca(tmp_path):
    from orca_assistant.orcafind import OrcaSearch
    rep = doctor(OrcaSearch("/opt/orca/orca", "real", "54 orca_* programs next to it",
                            ["/usr/bin/orca: a script with no orca_* programs"]))
    assert rep.ok
    t = rep.text
    assert "/opt/orca/orca  (confirmed: 54 orca_* programs next to it)" in t
    assert "skipped: /usr/bin/orca" in t and "matplotlib" in t and "3.11.2" in t
    assert "anthropic   : not installed (optional" in t
    assert "4 cores, 6.7 GiB total, 3.0 GiB available" in t and "nprocs=" in t
    assert "sbatch      : not found" in t and "bash        : /usr/bin/bash" in t


def test_doctor_report_without_orca():
    from orca_assistant.orcafind import OrcaSearch
    rep = doctor(OrcaSearch(None, "none", skipped=["/usr/bin/orca: screen reader"]))
    assert not rep.ok and "NOT FOUND" in rep.text and ENV_OVERRIDE in rep.text


def test_doctor_marks_unconfirmed_and_user_choices():
    from orca_assistant.orcafind import OrcaSearch
    assert "NOT confirmed" in doctor(OrcaSearch("/x/orca", "unconfirmed", "no helpers")).text
    assert "chosen by you" in doctor(OrcaSearch("/x/orca", "user", "set by X")).text


def test_path_order_is_respected_for_a_non_screen_reader(tmp_path):
    # e.g. a site wrapper script (no helpers next to it) placed first on purpose
    wrapper = make(tmp_path / "wrap", "orca", content=b"#!/bin/sh\nexec /opt/orca/orca \"$@\"\n")
    real_install(tmp_path / "qc")
    res = find_orca(env(tmp_path / "wrap", tmp_path / "qc"))
    assert res.path == str(wrapper) and res.verdict == "unconfirmed"
    assert any("another ORCA later in PATH" in s for s in res.skipped)


def test_screen_reader_is_skipped_wherever_it_is(tmp_path):
    real = real_install(tmp_path / "qc")
    screen_reader(tmp_path / "usr_bin")
    for order in (("qc", "usr_bin"), ("usr_bin", "qc")):
        res = find_orca(env(*(tmp_path / d for d in order)))
        assert res.path == str(real)


def test_a_script_that_merely_mentions_orca_is_not_the_screen_reader(tmp_path):
    p = make(tmp_path / "x", "orca", content=b"#!/bin/sh\n# runs orca from the module\nexec orca_real \"$@\"\n")
    assert classify(p)[0] == "unconfirmed"
