import json

from orca_assistant.inp import ERROR, INFO, WARNING, parse_inp
from orca_assistant.render import render_input
from orca_assistant.resources import Hardware
from orca_assistant.spec import CalcSpec

WATER = (
    "! B3LYP def2-TZVP D3BJ def2/J RIJCOSX TightSCF Opt Freq\n"
    "%pal nprocs 4 end\n"
    "%maxcore 2000\n"
    "* xyz 0 1\n"
    "O 0.000000 0.000000 0.117000\n"
    "H 0.000000 0.757000 -0.469000\n"
    "H 0.000000 -0.757000 -0.469000\n"
    "*\n"
)

SCAN = (
    "! B3LYP def2-SVP Opt Scan\n"
    "%geom Scan\n"
    "  B 0 1 = 0.8, 1.4, 7\n"
    "  end\n"
    "end\n"
    "* xyz 0 1\n"
    "O 0 0 0.117\n"
    "H 0 0.757 -0.469\n"
    "H 0 -0.757 -0.469\n"
    "*\n"
)


def levels(report):
    return [i.level for i in report.issues]


def messages(report):
    return " | ".join(i.message for i in report.issues)


def test_rendered_input_round_trips(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({
        "molecule": {"atoms": [
            {"element": "O", "x": 0, "y": 0, "z": 0.117},
            {"element": "H", "x": 0, "y": 0.757, "z": -0.469},
            {"element": "H", "x": 0, "y": -0.757, "z": -0.469}]},
        "resources": {"nprocs": 6, "maxcore_mb": 1500}}))
    spec = CalcSpec.model_validate_json(spec_path.read_text())
    report = parse_inp(render_input(spec))
    assert report.issues == []
    assert report.exit_code() == 0
    inp = report.inp
    assert (inp.nprocs, inp.maxcore_mb) == (6, 1500)
    assert (inp.charge, inp.multiplicity) == (0, 1)
    assert len(inp.atoms) == 3


def test_scan_block_is_read():
    report = parse_inp(SCAN)
    assert report.issues == []
    scan = report.inp.scan[0]
    assert (scan.kind, scan.atoms, scan.values) == ("B", [0, 1], [0.8, 1.4, 7.0])
    assert scan.line_no == 3


def test_scan_index_out_of_range():
    text = SCAN.replace("B 0 1 = 0.8, 1.4, 7", "B 0 5 = 0.8, 1.4, 7")
    report = parse_inp(text)
    assert report.exit_code() == 2
    assert "out of range for 3 atoms" in messages(report)


def test_scan_kind_needs_its_indices():
    text = SCAN.replace("B 0 1 =", "A 0 1 =")
    report = parse_inp(text)
    assert "scan A needs 3 atom indices" in messages(report)


def test_scan_needs_start_and_end():
    text = SCAN.replace("= 0.8, 1.4, 7", "= 0.8")
    report = parse_inp(text)
    assert "at least start and end" in messages(report)


def test_scan_steps_must_be_a_whole_number():
    text = SCAN.replace("= 0.8, 1.4, 7", "= 0.8, 1.4, 2.5")
    report = parse_inp(text)
    assert "whole number >= 2" in messages(report)


def test_scan_requested_without_definition():
    text = "! B3LYP Opt Scan\n* xyz 0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n*\n"
    report = parse_inp(text)
    assert any(i.level == WARNING and "no scan definition" in i.message
               for i in report.issues)


def test_unterminated_coordinate_block():
    report = parse_inp("! B3LYP\n* xyz 0 1\nO 0 0 0\nH 0 0 1\nH 0 1 0\n")
    assert report.exit_code() == 2
    assert "never closed" in messages(report)


def test_coordinate_header_needs_charge_and_multiplicity():
    report = parse_inp("! B3LYP\n* xyz\nO 0 0 0\n*\n")
    assert "needs charge and multiplicity" in messages(report)


def test_charge_multiplicity_parity_is_checked():
    report = parse_inp("! B3LYP\n* xyz 0 2\nO 0 0 0\nH 0 0 1\nH 0 1 0\n*\n")
    assert report.exit_code() == 2
    assert "multiplicity" in messages(report).lower()


def test_unknown_element_in_coordinates():
    report = parse_inp("! B3LYP\n* xyz 0 1\nQq 0 0 0\n*\n")
    assert "unknown element symbol" in messages(report)


def test_missing_simple_line():
    report = parse_inp("* xyz 0 1\nO 0 0 0\n*\n")
    assert report.exit_code() == 2
    assert "no simple-input line" in messages(report)


def test_pal_needs_nprocs():
    report = parse_inp(WATER.replace("%pal nprocs 4 end", "%pal end"))
    assert "%pal needs 'nprocs N'" in messages(report)


def test_maxcore_needs_a_value():
    report = parse_inp(WATER.replace("%maxcore 2000", "%maxcore"))
    assert "%maxcore needs a value in MB" in messages(report)


def test_unclosed_block_is_an_error():
    report = parse_inp("! B3LYP\n%geom\n  Scan\n  B 0 1 = 1.0, 1.5, 5\n* xyz 0 1\n"
                       "O 0 0 0\nH 0 0 1\n*\n")
    assert any(i.level == ERROR and "never closed" in i.message for i in report.issues)


def test_unchecked_block_is_reported():
    report = parse_inp(WATER.replace("%maxcore 2000", "%maxcore 2000\n%scf maxiter 300 end"))
    assert "%scf is not checked" in messages(report)
    assert "scf" in report.inp.unchecked


def test_duplicate_maxcore_warns():
    report = parse_inp(WATER.replace("%maxcore 2000", "%maxcore 2000\n%maxcore 3000"))
    assert "given more than once" in messages(report)
    assert report.inp.maxcore_mb == 3000


def test_double_counted_dispersion_from_keywords():
    text = WATER.replace("! B3LYP def2-TZVP D3BJ", "! wB97X-D3 def2-TZVP D3BJ")
    report = parse_inp(text)
    assert "double-counts it" in messages(report)


def test_composite_method_with_explicit_dispersion():
    text = WATER.replace("! B3LYP def2-TZVP D3BJ def2/J RIJCOSX",
                         "! r2SCAN-3c D4")
    report = parse_inp(text)
    assert "double-counts it" in messages(report)


def test_hardware_check_uses_parsed_resources():
    hw = Hardware(cores=2, ram_mb=4096, available_mb=None)
    report = parse_inp(WATER, hardware=hw)
    assert any(i.level == WARNING and "exceeds the 2 physical cores" in i.message
               for i in report.issues)


def test_without_hardware_no_hardware_issues():
    assert parse_inp(WATER).issues == []


def test_exit_codes_clean_warning_error():
    assert parse_inp(WATER).exit_code() == 0
    assert parse_inp(WATER.replace("%maxcore 2000", "%maxcore 2000\n%maxcore 3000")).exit_code() == 1
    assert parse_inp("! B3LYP\n").exit_code() == 2


def test_json_report_has_no_atom_coordinates():
    payload = json.loads(json.dumps(parse_inp(WATER).to_dict()))
    assert payload["n_atoms"] == 3
    assert payload["keywords"][:2] == ["B3LYP", "def2-TZVP"]
    assert payload["issues"] == []
    assert "atoms" not in payload


def test_molecule_property_is_validated():
    assert parse_inp(WATER).inp.molecule is not None
    broken = parse_inp("! B3LYP\n* xyz 0 2\nO 0 0 0\nH 0 0 1\nH 0 1 0\n*\n")
    assert broken.inp.molecule is None


def test_info_level_does_not_fail_the_check():
    report = parse_inp(WATER.replace("%maxcore 2000",
                                     "%maxcore 2000\n%scf maxiter 300 end"))
    assert INFO in levels(report)
    assert report.exit_code() == 0
