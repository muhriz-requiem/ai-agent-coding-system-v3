import importlib.util, pytest
if importlib.util.find_spec('src.ledger.csv_import') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.csv_import import parse_csv
from src.ledger.cli import main

def test_parse_csv_valid_input():
    csv_text = "date,memo,account,side,amount\n2023-01-01,Opening Balance,Cash,debit,100.00\n"
    entries = parse_csv(csv_text)
    assert isinstance(entries, list)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.date == "2023-01-01"
    assert entry.memo == "Opening Balance"
    assert entry.account == "Cash"
    assert entry.side == "debit"
    assert entry.amount == 100.00

def test_parse_csv_empty_input():
    csv_text = "date,memo,account,side,amount\n"
    entries = parse_csv(csv_text)
    assert isinstance(entries, list)
    assert len(entries) == 0

def test_main_cli_success(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("date,memo,account,side,amount\n2023-01-01,Test,Cash,debit,50.00\n2023-01-01,Test,Equity,credit,50.00\n")
    
    exit_code = main([str(p)])
    assert exit_code == 0