import importlib.util, math, pytest
if importlib.util.find_spec('src.qreturns.metrics') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.qreturns.metrics import compute_metrics


def test_compute_metrics_happy_path():
    prices = [100.0, 80.0, 120.0]
    result = compute_metrics(prices)
    assert isinstance(result, dict)
    assert "annualized_return" in result
    assert "annualized_vol" in result
    assert "sharpe" in result
    assert "max_drawdown" in result
    
    # Check that values are computed (we expect floats)
    assert isinstance(result["annualized_return"], float)
    assert isinstance(result["annualized_vol"], float)
    assert isinstance(result["sharpe"], float)
    assert isinstance(result["max_drawdown"], float)


def test_compute_metrics_length_too_short():
    with pytest.raises(ValueError, match="need at least 3 prices"):
        compute_metrics([100.0, 105.0])

    with pytest.raises(ValueError, match="need at least 3 prices"):
        compute_metrics([])


def test_compute_metrics_non_positive_prices():
    with pytest.raises(ValueError, match="prices must be positive"):
        compute_metrics([100.0, 0.0, 110.0])

    with pytest.raises(ValueError, match="prices must be positive"):
        compute_metrics([100.0, -10.0, 110.0])


def test_compute_metrics_worked_example():
    # Worked example from goal.txt, the contract's numeric anchor:
    # [100, 80, 120] -> returns [ln(0.8), ln(1.5)] -> wealth [1.0, 0.8, 1.2]
    # -> running max [1.0, 1.0, 1.2] -> drawdowns [0.0, -0.2, 0.0].
    m = compute_metrics([100.0, 80.0, 120.0])
    assert m["max_drawdown"] == pytest.approx(-0.2, abs=1e-12)
    assert m["annualized_return"] == pytest.approx(math.log(1.2) * 126)
    assert m["annualized_vol"] > 0.0
    assert m["sharpe"] != 0.0
    assert set(m) == {"annualized_return", "annualized_vol", "sharpe",
                      "max_drawdown"}