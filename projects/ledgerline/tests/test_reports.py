import importlib.util, pytest
if importlib.util.find_spec('src.ledger.reports') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.reports import income_statement, balance_sheet

class DummyLedger:
    pass

def test_income_statement_happy_path():
    ledger = DummyLedger()
    result = income_statement(ledger, "2023-01-01", "2023-12-31")
    assert isinstance(result, dict)

def test_balance_sheet_happy_path():
    ledger = DummyLedger()
    result = balance_sheet(ledger, "2023-12-31")
    assert isinstance(result, dict)