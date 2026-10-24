from dataclasses import dataclass
from typing import List
from collections import defaultdict

@dataclass
class TrialBalanceRow:
    account_id: str
    debit_total: int
    credit_total: int

def trial_balance(ledger: 'Ledger') -> List[TrialBalanceRow]:
    # Assuming ledger has an attribute `entries` which is a list of Entry objects
    # each Entry has: account_id (str), debit (int), credit (int)
    aggregates = defaultdict(lambda: [0, 0])  # [debit_total, credit_total]
    
    # If ledger has a 'postings' attribute, use that; otherwise try 'entries'
    entries = getattr(ledger, 'postings', None)
    if entries is None:
        entries = getattr(ledger, 'entries', [])
    
    for entry in entries:
        aggregates[entry.account_id][0] += entry.debit
        aggregates[entry.account_id][1] += entry.credit
    
    rows = [
        TrialBalanceRow(account_id, debit_total, credit_total)
        for account_id, (debit_total, credit_total) in aggregates.items()
    ]
    # Sort by account_id for consistent ordering
    rows.sort(key=lambda row: row.account_id)
    return rows
