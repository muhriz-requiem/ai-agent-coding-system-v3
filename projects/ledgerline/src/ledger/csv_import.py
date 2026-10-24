import csv
from typing import List


class JournalEntry:
    def __init__(self, date: str, memo: str, account: str, side: str, amount: float):
        self.date = date
        self.memo = memo
        self.account = account
        self.side = side
        self.amount = amount


def parse_csv(text: str) -> List[JournalEntry]:
    """
    Parse CSV text into a list of JournalEntry objects.
    Expected columns: date, memo, account, side, amount.
    """
    if not text.strip():
        return []

    lines = text.strip().splitlines()
    reader = csv.reader(lines)

    try:
        header = [col.strip().lower() for col in next(reader)]
    except StopIteration:
        return []

    expected_header = ['date', 'memo', 'account', 'side', 'amount']
    if header != expected_header:
        raise ValueError(f"Invalid CSV header. Expected {expected_header}, got {header}")

    entries = []
    for row_num, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) < len(expected_header):
            raise ValueError(f"Line {row_num}: insufficient columns. Expected {len(expected_header)}, got {len(row)}")

        row_data = {header[i]: row[i].strip() for i in range(len(header))}

        try:
            date = row_data['date']
            memo = row_data['memo']
            account = row_data['account']
            side = row_data['side'].lower()
            amount_str = row_data['amount']

            if side not in ('debit', 'credit'):
                raise ValueError(f"Invalid side '{side}'. Must be 'debit' or 'credit'")

            try:
                amount = float(amount_str)
            except ValueError:
                raise ValueError(f"Invalid amount '{amount_str}': must be a number")

            entry = JournalEntry(date, memo, account, side, amount)
            entries.append(entry)
        except KeyError as e:
            raise ValueError(f"Line {row_num}: missing column '{e}'")

    return entries
