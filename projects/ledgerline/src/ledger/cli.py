import sys
from typing import List
from src.ledger.csv_import import parse_csv, JournalEntry


def main(argv: List[str]) -> int:
    if len(argv) != 1:
        print("Usage: ledger-cli <csv_file>", file=sys.stderr)
        return 1

    file_path = argv[0]
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            csv_text = f.read()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return 1

    try:
        entries = parse_csv(csv_text)
    except Exception as e:
        print(f"Error parsing CSV: {e}", file=sys.stderr)
        return 1

    if not entries:
        print("No entries found in CSV.")
        return 0

    accounts = {}
    for entry in entries:
        account = entry.account
        amount = entry.amount
        if account not in accounts:
            accounts[account] = {'debit': 0.0, 'credit': 0.0}
        if entry.side.lower() == 'debit':
            accounts[account]['debit'] += amount
        elif entry.side.lower() == 'credit':
            accounts[account]['credit'] += amount
        else:
            print(f"Warning: Unknown side '{entry.side}' for entry", file=sys.stderr)

    print("Trial Balance:")
    print("-" * 50)
    print(f"{'Account':<30} {'Debit':>10} {'Credit':>10}")
    print("-" * 50)
    total_debit = 0.0
    total_credit = 0.0
    for account in sorted(accounts.keys()):
        debit = accounts[account]['debit']
        credit = accounts[account]['credit']
        total_debit += debit
        total_credit += credit
        print(f"{account:<30} {debit:>10.2f} {credit:>10.2f}")
    print("-" * 50)
    print(f"{'Total':<30} {total_debit:>10.2f} {total_credit:>10.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
