from typing import Dict, List, Tuple, Optional


class JournalEntry:
    """Represents a journal entry containing debit and credit line items."""

    def __init__(
        self,
        lines: Optional[List[Tuple[str, int, int]]] = None,
        description: str = "",
    ):
        """
        Initialize a JournalEntry.

        Args:
            lines: A list of tuples in the format (account_id, debit_amount, credit_amount).
                   Defaults to an empty list.
            description: Optional text describing the entry.
        """
        self.lines: List[Tuple[str, int, int]] = lines if lines is not None else []
        self.description: str = description

    def add_line(self, account_id: str, debit: int = 0, credit: int = 0) -> None:
        """Add a line item to the journal entry."""
        self.lines.append((account_id, debit, credit))


class Ledger:
    """Main ledger engine for recording entries and tracking account balances."""

    def __init__(self):
        self.entries: List[JournalEntry] = []

    def add_entry(self, entry: JournalEntry) -> None:
        """Add a journal entry to the ledger."""
        self.entries.append(entry)

    def balances(self) -> Dict[str, int]:
        """
        Calculate and return the net balance for all accounts in the ledger.

        Net balance is defined as total debits minus total credits for each account.
        """
        acc_balances: Dict[str, int] = {}
        for entry in self.entries:
            for account_id, debit, credit in entry.lines:
                if account_id not in acc_balances:
                    acc_balances[account_id] = 0
                acc_balances[account_id] += debit - credit
        return acc_balances

    def balance_of(self, account_id: str) -> int:
        """Return the net balance of a specific account."""
        return self.balances().get(account_id, 0)

    def is_balanced(self) -> bool:
        """
        Check if the ledger as a whole is balanced.

        The ledger is balanced if the sum of all debits across all entries equals
        the sum of all credits across all entries.
        """
        total_debit = 0
        total_credit = 0
        for entry in self.entries:
            for _, debit, credit in entry.lines:
                total_debit += debit
                total_credit += credit
        return total_debit == total_credit
