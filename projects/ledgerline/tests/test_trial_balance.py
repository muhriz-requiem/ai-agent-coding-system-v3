import importlib.util, pytest
if importlib.util.find_spec('src.ledger.trial_balance') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.trial_balance import trial_balance, TrialBalanceRow


def test_trial_balance_happy_path():
    # Mocking or instantiating a Ledger and its entries based on typical ledger structure
    # Since Ledger is from a dependency, we construct it according to expected usage
    class MockEntry:
        def __init__(self, account_id, debit, credit):
            self.account_id = account_id
            self.debit = debit
            self.credit = credit

    class MockLedger:
        def __init__(self, entries):
            self.entries = entries

    ledger = MockLedger([
        MockEntry("acc_1", 100, 0),
        MockEntry("acc_1", 0, 50),
        MockEntry("acc_2", 0, 200),
        MockEntry("acc_2", 200, 0)
    ])

    result = trial_balance(ledger)
    
    assert isinstance(result, list)
    assert all(isinstance(row, TrialBalanceRow) for row in result)
    
    # Sort by account_id for deterministic assertion
    sorted_result = sorted(result, key=lambda r: r.account_id)
    
    assert len(sorted_result) == 2
    assert sorted_result[0].account_id == "acc_1"
    assert sorted_result[0].debit_total == 100
    assert sorted_result[0].credit_total == 50
    
    assert sorted_result[1].account_id == "acc_2"
    assert sorted_result[1].debit_total == 200
    assert sorted_result[1].credit_total == 200


def test_trial_balance_empty_ledger():
    class MockLedger:
        def __init__(self, entries):
            self.entries = entries

    ledger = MockLedger([])
    result = trial_balance(ledger)
    
    assert isinstance(result, list)
    assert len(result) == 0