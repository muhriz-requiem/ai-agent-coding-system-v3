from enum import Enum
from dataclasses import dataclass

class AccountType(Enum):
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    EQUITY = "EQUITY"
    REVENUE = "REVENUE"
    EXPENSE = "EXPENSE"

@dataclass
class Account:
    id: str
    name: str
    type: AccountType

def normal_balance(account_type: AccountType) -> str:
    if account_type in [AccountType.ASSET, AccountType.EXPENSE]:
        return "DEBIT"
    elif account_type in [AccountType.LIABILITY, AccountType.EQUITY, AccountType.REVENUE]:
        return "CREDIT"
    else:
        raise ValueError(f"Unknown account type: {account_type}")
