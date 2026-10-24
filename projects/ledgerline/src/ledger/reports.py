from __future__ import annotations

from typing import Dict


def income_statement(ledger: "Ledger", start: str, end: str) -> Dict[str, float]:
    """
    Calculate the income statement for the period [start, end].

    Parameters
    ----------
    ledger : Ledger
        The ledger containing transactions and accounts.
    start : str
        Start date in ISO format (inclusive).
    end : str
        End date in ISO format (inclusive).

    Returns
    -------
    dict
        Dictionary with keys ``revenues`` and ``expenses`` representing the
        total amounts for each category.
    """
    revenues = 0.0
    expenses = 0.0

    for txn in getattr(ledger, "transactions", []):
        txn_date = getattr(txn, "date", None)
        if txn_date is None:
            continue
        if not (start <= txn_date <= end):
            continue

        amount = getattr(txn, "amount", 0.0)
        account = getattr(txn, "account", None)
        if account is None:
            continue
        acct_type = getattr(account, "type", "").strip().lower()
        if acct_type == "revenue":
            revenues += amount
        elif acct_type == "expense":
            expenses += amount

    return {"revenues": revenues, "expenses": expenses}


def balance_sheet(ledger: "Ledger", as_of: str) -> Dict[str, float]:
    """
    Calculate the balance sheet as of a given date.

    Parameters
    ----------
    ledger : Ledger
        The ledger containing transactions and accounts.
    as_of : str
        Date in ISO format (inclusive) for snapshot.

    Returns
    -------
    dict
        Dictionary with keys ``assets``, ``liabilities``, and ``equity``.
    """
    assets = 0.0
    liabilities = 0.0
    equity = 0.0

    for txn in getattr(ledger, "transactions", []):
        txn_date = getattr(txn, "date", None)
        if txn_date is None or txn_date > as_of:
            continue
        amount = getattr(txn, "amount", 0.0)
        account = getattr(txn, "account", None)
        if account is None:
            continue
        acct_type = getattr(account, "type", "").strip().lower()
        if acct_type == "asset":
            assets += amount
        elif acct_type == "liability":
            liabilities += amount
        elif acct_type == "equity":
            equity += amount
        # Revenue and expense accounts are not included in the balance sheet.

    return {"assets": assets, "liabilities": liabilities, "equity": equity}
