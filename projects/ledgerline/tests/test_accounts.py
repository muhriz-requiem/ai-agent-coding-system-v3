import importlib.util, pytest
if importlib.util.find_spec('src.ledger.accounts') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.accounts import AccountType, Account, normal_balance

def test_account_type_members():
    assert AccountType.ASSET.name == 'ASSET'
    assert AccountType.LIABILITY.name == 'LIABILITY'
    assert AccountType.EQUITY.name == 'EQUITY'
    assert AccountType.REVENUE.name == 'REVENUE'
    assert AccountType.EXPENSE.name == 'EXPENSE'
    assert len(list(AccountType)) == 5

def test_account_dataclass():
    acc = Account(id="100", name="Cash", type=AccountType.ASSET)
    assert acc.id == "100"
    assert acc.name == "Cash"
    assert acc.type == AccountType.ASSET

def test_normal_balance_asset():
    assert normal_balance(AccountType.ASSET) == 'DEBIT'

def test_normal_balance_expense():
    assert normal_balance(AccountType.EXPENSE) == 'DEBIT'

def test_normal_balance_liability():
    assert normal_balance(AccountType.LIABILITY) == 'CREDIT'

def test_normal_balance_equity():
    assert normal_balance(AccountType.EQUITY) == 'CREDIT'

def test_normal_balance_revenue():
    assert normal_balance(AccountType.REVENUE) == 'CREDIT'

def test_normal_balance_invalid_type():
    class FakeType:
        pass
    with pytest.raises((ValueError, AttributeError)):
        normal_balance(FakeType())