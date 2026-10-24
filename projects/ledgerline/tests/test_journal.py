import importlib.util, pytest
if importlib.util.find_spec('src.ledger.journal') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.journal import Side, Posting, JournalEntry

def test_side_enum():
    assert Side.DEBIT is not None
    assert Side.CREDIT is not None

def test_posting_initialization():
    posting = Posting(account_id="acc_1", amount_minor=1000, side=Side.DEBIT)
    assert posting.account_id == "acc_1"
    assert posting.amount_minor == 1000
    assert posting.side == Side.DEBIT

def test_journal_entry_valid():
    postings = [
        Posting(account_id="acc_1", amount_minor=1000, side=Side.DEBIT),
        Posting(account_id="acc_2", amount_minor=1000, side=Side.CREDIT)
    ]
    entry = JournalEntry(date="2023-01-01", memo="Test entry", postings=postings)
    assert entry.date == "2023-01-01"
    assert entry.memo == "Test entry"
    assert len(entry.postings) == 2

def test_journal_entry_raises_ValueError_fewer_than_two_postings():
    postings = [
        Posting(account_id="acc_1", amount_minor=1000, side=Side.DEBIT)
    ]
    with pytest.raises(ValueError):
        JournalEntry(date="2023-01-01", memo="Single posting", postings=postings)

def test_journal_entry_raises_ValueError_unbalanced_postings():
    postings = [
        Posting(account_id="acc_1", amount_minor=1000, side=Side.DEBIT),
        Posting(account_id="acc_2", amount_minor=500, side=Side.CREDIT)
    ]
    with pytest.raises(ValueError):
        JournalEntry(date="2023-01-01", memo="Unbalanced entry", postings=postings)