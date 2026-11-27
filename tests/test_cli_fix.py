from orca_assistant.cli import main

EQ = "O 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469"
INP = f"! B3LYP def2-SVP\n%scf MaxIter 2 end\n* xyz 0 1\n{EQ}\n*\n"
OUT = "SCF NOT CONVERGED AFTER   1 CYCLES\nORCA finished by error termination in LEANSCF\n"


def setup(tmp_path, inp=INP, out=OUT):
    (tmp_path / "j.inp").write_text(inp)
    (tmp_path / "j.out").write_text(out)
    return tmp_path / "j.out"


def test_fix_writes_a_new_input_and_keeps_the_original(tmp_path, capsys):
    out = setup(tmp_path)
    assert main(["fix", str(out)]) == 0
    assert (tmp_path / "j.inp").read_text() == INP
    new = (tmp_path / "j_fix1.inp").read_text()
    assert "SlowConv" in new and "MaxIter 500" in new
    assert "+%scf MaxIter 500 end" in capsys.readouterr().out


def test_second_fix_does_not_overwrite_the_first(tmp_path):
    out = setup(tmp_path)
    assert main(["fix", str(out)]) == 0 and main(["fix", str(out)]) == 0
    assert (tmp_path / "j_fix1.inp").exists() and (tmp_path / "j_fix2.inp").exists()


def test_fix_refuses_to_overwrite_an_explicit_output(tmp_path, capsys):
    out = setup(tmp_path)
    target = tmp_path / "keep.inp"
    target.write_text("precious")
    assert main(["fix", str(out), "-o", str(target)]) == 1
    assert target.read_text() == "precious" and "already exists" in capsys.readouterr().err


def test_nothing_to_fix(tmp_path, capsys):
    out = setup(tmp_path, out="****ORCA TERMINATED NORMALLY****\n")
    assert main(["fix", str(out)]) == 1
    assert "nothing to fix" in capsys.readouterr().out
    assert not (tmp_path / "j_fix1.inp").exists()


def test_fix_multiplicity_option(tmp_path):
    mult_out = ("[file orca_main/main_util_tools.cpp, line 754]: Error : multiplicity (2) is "
                "even and number of electrons (10) is even -> impossible\n")
    out = setup(tmp_path, inp=INP.replace("0 1", "0 2"), out=mult_out)
    assert main(["fix", str(out)]) == 1                     # not changed without the option
    assert main(["fix", str(out), "--multiplicity", "1"]) == 0
    assert "* xyz 0 1" in (tmp_path / "j_fix1.inp").read_text()


def test_fix_missing_files(tmp_path, capsys):
    assert main(["fix", str(tmp_path / "no.out")]) == 3
    (tmp_path / "x.out").write_text(OUT)
    assert main(["fix", str(tmp_path / "x.out")]) == 3
    assert "use --inp" in capsys.readouterr().err
