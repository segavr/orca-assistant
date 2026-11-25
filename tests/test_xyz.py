import pytest

from orca_assistant.spec import atoms_from_xyz

WATER_STD = (
    "3\n"
    "water, B3LYP/def2-SVP\n"
    "O 0.000000 0.000000 0.117000\n"
    "H 0.000000 0.757000 -0.469000\n"
    "H 0.000000 -0.757000 -0.469000\n"
)


def test_standard_xyz_keeps_all_atoms():
    atoms = atoms_from_xyz(WATER_STD)
    assert [a.element for a in atoms] == ["O", "H", "H"]
    assert atoms[0].z == pytest.approx(0.117)


def test_bare_atom_lines_without_header():
    atoms = atoms_from_xyz("O 0 0 0.117\nH 0 0.757 -0.469\n")
    assert [a.element for a in atoms] == ["O", "H"]


def test_truncated_file_is_an_error():
    text = "5\nwater\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    with pytest.raises(ValueError, match="declares 5 atoms"):
        atoms_from_xyz(text)


def test_missing_comment_line_is_accepted():
    text = "3\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    assert len(atoms_from_xyz(text)) == 3


def test_four_word_comment_is_not_mistaken_for_an_atom():
    text = "3\nopt freq b3lyp def2\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n"
    assert [a.element for a in atoms_from_xyz(text)] == ["O", "H", "H"]


def test_short_coordinate_line_names_the_line():
    text = "2\nwater\nO 0 0 0.117\nH 0 0.757\n"
    with pytest.raises(ValueError, match="line 4"):
        atoms_from_xyz(text)


def test_non_numeric_coordinate_names_the_line():
    text = "2\nwater\nO 0 0 0.117\nH 0 0.757 abc\n"
    with pytest.raises(ValueError, match="line 4"):
        atoms_from_xyz(text)


def test_unknown_element_names_the_line():
    text = "2\nwater\nO 0 0 0.117\nXx 0 0.757 0.0\n"
    with pytest.raises(ValueError, match="Xx"):
        atoms_from_xyz(text)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError, match="no atoms"):
        atoms_from_xyz("   \n\n")


def test_zero_atoms_in_header_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        atoms_from_xyz("0\nnothing here\n")
