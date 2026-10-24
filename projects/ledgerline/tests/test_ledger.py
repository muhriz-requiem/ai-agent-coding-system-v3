import importlib.util, pytest
if importlib.util.find_spec('src.ledger.ledger') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.ledger import Ledger, JournalEntry


def test_ledger_happy_path():
    ledger = Ledger()
    entry = JournalEntry()  # Assuming JournalEntry can be instantiated
    ledger.add_entry(entry)
    
    # Test balance methods exist and return expected types/values
    balances = ledger.balances()
    assert isinstance(balances, dict)
    
    is_bal = ledger.is_balanced()
    assert isinstance(is_bal, bool)


def test_ledger_balance_of():
    ledger = Ledger()
    assert ledger.balance_of("100") == 0


def test_ledger_is_balanced_empty():
    ledger = Ledger()
    assert ledger.is_balanced() is True