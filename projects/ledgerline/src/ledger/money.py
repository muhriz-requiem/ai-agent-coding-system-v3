"""
Money formatting and parsing utilities.

Provides functions to convert between decimal string representations
and integer minor units (cents) for precise monetary calculations.
"""

from decimal import Decimal, InvalidOperation


def parse_amount(text: str) -> int:
    """
    Parse a decimal string amount into integer minor units (cents).

    Args:
        text: A string representation of a monetary amount (e.g., "10.50", "-5.00", "0.05")

    Returns:
        Integer minor units (e.g., 1050 for "10.50", -500 for "-5.00")

    Raises:
        ValueError: If the input is malformed, empty, or contains invalid characters.
    """
    if not text or not text.strip():
        raise ValueError("Empty or whitespace input is not a valid amount")

    stripped = text.strip()

    try:
        # Use Decimal for precise parsing
        amount = Decimal(stripped)
    except InvalidOperation:
        raise ValueError(f"Invalid amount format: '{text}'")

    # Check for more than 2 decimal places
    if amount.as_tuple().exponent < -2:
        raise ValueError(f"Amount has more than 2 decimal places: '{text}'")

    # Convert to minor units (multiply by 100)
    minor_units = int(amount * 100)

    return minor_units


def format_amount(minor_units: int) -> str:
    """
    Convert integer minor units to a decimal string representation.

    Args:
        minor_units: Integer amount in minor units (cents)

    Returns:
        String representation with exactly 2 decimal places (e.g., "10.50", "-5.00", "0.05")
    """
    # Handle negative values
    is_negative = minor_units < 0
    abs_units = abs(minor_units)

    # Calculate dollars and cents
    dollars = abs_units // 100
    cents = abs_units % 100

    # Format with sign
    sign = "-" if is_negative else ""
    return f"{sign}{dollars}.{cents:02d}"
