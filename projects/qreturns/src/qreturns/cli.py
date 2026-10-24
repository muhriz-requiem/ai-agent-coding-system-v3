"""CLI module for qreturns."""

import csv
import sys

from .metrics import compute_metrics


def main(argv: list[str]) -> int:
    """Read prices from a CSV file and print computed metrics.

    Args:
        argv: Argument list where argv[0] is the path to the CSV file.

    Returns:
        0 on success, 2 on invalid input.
    """
    if not argv:
        print("bad csv: missing file path", file=sys.stderr)
        return 2

    filepath = argv[0]

    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                print("bad csv: empty file", file=sys.stderr)
                return 2

            prices = []
            header_lower = [h.strip().lower() for h in header]

            if "price" in header_lower:
                price_idx = header_lower.index("price")
                for row in reader:
                    if not row:
                        continue
                    try:
                        prices.append(float(row[price_idx]))
                    except (ValueError, IndexError):
                        print("bad csv: invalid price value", file=sys.stderr)
                        return 2
            else:
                try:
                    prices.append(float(header[0]))
                except (ValueError, IndexError):
                    print("bad csv: invalid price value", file=sys.stderr)
                    return 2
                for row in reader:
                    if not row:
                        continue
                    try:
                        prices.append(float(row[0]))
                    except (ValueError, IndexError):
                        print("bad csv: invalid price value", file=sys.stderr)
                        return 2

            if not prices:
                print("bad csv: no price data", file=sys.stderr)
                return 2

            metrics = compute_metrics(prices)
            for key, value in metrics.items():
                print(f"{key}={value}")
            return 0

    except FileNotFoundError:
        print(f"bad csv: file not found: {filepath}", file=sys.stderr)
        return 2
    except PermissionError:
        print(f"bad csv: permission denied: {filepath}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"bad csv: {e}", file=sys.stderr)
        return 2
