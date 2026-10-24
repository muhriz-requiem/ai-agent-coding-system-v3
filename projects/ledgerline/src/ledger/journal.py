from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List


class Side(Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


@dataclass
class Posting:
    account_id: str
    amount_minor: int
    side: Side


@dataclass
class JournalEntry:
    date: str
    memo: str
    postings: List[Posting]

    def __post_init__(self) -> None:
        if len(self.postings) < 2:
            raise ValueError("JournalEntry must have at least two postings")
        total_debit = sum(
            p.amount_minor for p in self.postings if p.side == Side.DEBIT
        )
        total_credit = sum(
            p.amount_minor for p in self.postings if p.side == Side.CREDIT
        )
        if total_debit != total_credit:
            raise ValueError("Total debits must equal total credits")
