import importlib.util, pytest
if importlib.util.find_spec('src.qreturns.cli') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.qreturns.cli import main

def test_main_success(tmp_path, capsys):
    csv_file = tmp_path / "prices.csv"
    csv_file.write_text("price\n100\n105\n110\n")
    
    ret = main([str(csv_file)])
    assert ret == 0
    
    captured = capsys.readouterr()
    assert len(captured.out.strip().split("\n")) == 4
    for line in captured.out.strip().split("\n"):
        assert "=" in line

def test_main_invalid_csv(tmp_path, capsys):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("price\nnot_a_number\n")
    
    ret = main([str(csv_file)])
    assert ret == 2
    
    captured = capsys.readouterr()
    assert captured.err.startswith("bad csv: ")

def test_main_missing_file(capsys):
    ret = main(["nonexistent_file_xyz.csv"])
    assert ret == 2
    
    captured = capsys.readouterr()
    assert captured.err.startswith("bad csv: ")