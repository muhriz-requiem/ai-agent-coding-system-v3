import math

def compute_metrics(prices: list[float]) -> dict:
    if len(prices) < 3:
        raise ValueError('need at least 3 prices')
    if any(p <= 0 for p in prices):
        raise ValueError('prices must be positive')
    
    log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
    n = len(log_returns)
    mean_return = sum(log_returns) / n
    annualized_return = mean_return * 252
    
    variance = sum((x - mean_return) ** 2 for x in log_returns) / (n - 1)
    vol = math.sqrt(variance)
    annualized_vol = vol * math.sqrt(252)
    
    if annualized_vol == 0:
        sharpe = 0.0
    else:
        sharpe = annualized_return / annualized_vol
    
    # Contract convention (goal.txt): wealth path w_t = price_t / price_0,
    # running max INCLUDES w_0, drawdown_t = w_t / running_max - 1 <= 0,
    # max_drawdown = min(all drawdowns).
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        if price > peak:
            peak = price
        dd = price / peak - 1.0
        if dd < max_dd:
            max_dd = dd
    
    return {
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd
    }
