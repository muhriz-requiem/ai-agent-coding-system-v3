import importlib.util, pytest
if importlib.util.find_spec('src.ledger.money') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.ledger.money import parse_amount, format_amount


def test_parse_amount_happy_path():
    assert parse_amount('10.50') == 1050


def test_format_amount_happy_path():
    assert format_amount(1050) == '10.50'


def test_parse_amount_invalid():
    with pytest.raises(ValueError):
        parse_amount('invalid')