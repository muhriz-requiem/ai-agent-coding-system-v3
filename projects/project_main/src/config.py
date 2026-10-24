import os
import sys
from pathlib import Path
from typing import List, Dict

try:
    from dotenv import load_dotenv
    if "pytest" not in sys.modules:
        load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean env var (0/false/no disables; empty keeps default)."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    return raw.strip().lower() not in ("false", "0", "no")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "")))
    except (TypeError, ValueError):
        return default


DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ", "AMD"]
DEFAULT_TIER_MODEL_MAP = {
    "tier1": "gpt-4o-mini",
    "tier2": "gpt-4o", 
    "tier3": "gpt-4.1"
}

class Settings:
    def __init__(self):
        self.alpaca_api_key = os.environ.get("ALPACA_API_KEY", "")
        self.alpaca_secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self.alpaca_base_url = os.environ.get("ALPACA_BASE_URL", "")
        
        # Parse PAPER_ONLY as boolean with default True
        paper_only_str = os.environ.get("PAPER_ONLY", "").lower()
        self.paper_only = paper_only_str not in ("false", "0", "no") if paper_only_str else True
        
        # Parse WATCHLIST from env or use default
        watchlist_env = os.environ.get("WATCHLIST", "")
        if watchlist_env:
            self.watchlist = [ticker.strip().upper() for ticker in watchlist_env.split(",") if ticker.strip()]
            if not self.watchlist:
                self.watchlist = DEFAULT_WATCHLIST
        else:
            self.watchlist = DEFAULT_WATCHLIST
            
        # Parse TIER_MODEL_MAP from env or use default
        tier_model_env = os.environ.get("TIER_MODEL_MAP", "")
        if tier_model_env:
            try:
                self.tier_model_map = {}
                for mapping in tier_model_env.split(","):
                    if ":" in mapping:
                        tier, model = mapping.split(":", 1)
                        tier = tier.strip()
                        model = model.strip()
                        if tier and model:
                            self.tier_model_map[tier] = model
                if not self.tier_model_map:
                    self.tier_model_map = DEFAULT_TIER_MODEL_MAP
            except:
                self.tier_model_map = DEFAULT_TIER_MODEL_MAP
        else:
            self.tier_model_map = DEFAULT_TIER_MODEL_MAP

        # Per-run cost guard: max tickers analyzed in one invocation
        self.max_tickers_per_run = _env_int("MAX_TICKERS_PER_RUN", 10)

        # ---- Whale/flow data sources (see .env.example for keys) ----
        self.edgar_identity = os.environ.get("EDGAR_IDENTITY", "")
        self.finra_api_key = os.environ.get("FINRA_API_KEY", "")
        self.securitiesdb_api_key = os.environ.get("SECURITIESDB_API_KEY", "")

        self.insider_enabled = _env_bool("INSIDER_ENABLED", True)
        self.institutional_enabled = _env_bool("INSTITUTIONAL_ENABLED", True)
        self.options_flow_enabled = _env_bool("OPTIONS_FLOW_ENABLED", True)
        self.dark_pool_enabled = _env_bool("DARK_POOL_ENABLED", True)

        self.uoa_min_vol_oi_ratio = _env_float("UOA_MIN_VOL_OI_RATIO", 3.0)
        self.uoa_min_premium = _env_float("UOA_MIN_PREMIUM", 25000.0)
        self.uoa_max_dte = _env_int("UOA_MAX_DTE", 45)
        self.insider_lookback_days = _env_int("INSIDER_LOOKBACK_DAYS", 90)
        self.cluster_insider_threshold = _env_int("CLUSTER_INSIDER_THRESHOLD", 3)
        self.flow_veto_min_holders = _env_int("FLOW_VETO_MIN_HOLDERS", 10)
        self.uoa_put_skew_veto = _env_float("UOA_PUT_SKEW_VETO", 0.30)

        # Portfolio sizing extras
        self.min_trade_notional = _env_float("MIN_TRADE_NOTIONAL", 10.0)
        self.min_trade_notional_pct = _env_float("MIN_TRADE_NOTIONAL_PCT", 0.01)
        self.overnight_stats_path = os.environ.get(
            "OVERNIGHT_STATS_PATH", "data/backtest_stats_overnight.json"
        )
        self.session_max_buys_per_ticker = _env_int("SESSION_MAX_BUYS_PER_TICKER", 1)
        self.session_max_sells_per_ticker = _env_int("SESSION_MAX_SELLS_PER_TICKER", 1)
        self.overnight_max_names = _env_int("OVERNIGHT_MAX_NAMES", 5)

        # Correlation clustering
        self.corr_clustering_enabled = _env_bool("CORR_CLUSTERING_ENABLED", True)

        # ---- Pairs trading (stat-arb module) ----
        self.pairs_enabled = _env_bool("PAIRS_ENABLED", True)
        self.universe_source = os.environ.get("UNIVERSE_SOURCE", "sp400")
        self.universe_min_adv = _env_float("UNIVERSE_MIN_ADV", 1_000_000.0)
        self.universe_min_mcap = _env_float("UNIVERSE_MIN_MCAP", 2_000_000_000.0)
        self.universe_max_mcap = _env_float("UNIVERSE_MAX_MCAP", 10_000_000_000.0)
        self.universe_refresh_days = _env_int("UNIVERSE_REFRESH_DAYS", 7)
        self.universe_fallback_path = os.environ.get(
            "UNIVERSE_FALLBACK_PATH", "data/universe_sp400_fallback.txt"
        )

        self.coint_lookback_days = _env_int("COINT_LOOKBACK_DAYS", 252)
        self.coint_adf_pvalue = _env_float("COINT_ADF_PVALUE", 0.05)
        self.coint_min_half_life = _env_float("COINT_MIN_HALF_LIFE", 5.0)
        self.coint_max_half_life = _env_float("COINT_MAX_HALF_LIFE", 60.0)
        self.coint_max_lag = _env_int("COINT_MAX_LAG", 1)
        self.pair_prescreen_corr = _env_float("PAIR_PRESCREEN_CORR", 0.70)
        self.pair_price_period = os.environ.get("PAIR_PRICE_PERIOD", "2y")

        self.max_pairs = _env_int("MAX_PAIRS", 10)
        self.max_pairs_per_sector = _env_int("MAX_PAIRS_PER_SECTOR", 3)
        self.min_pair_sharpe = _env_float("MIN_PAIR_SHARPE", 1.0)
        self.pair_rebalance_days = _env_int("PAIR_REBALANCE_DAYS", 21)

        self.pair_z_entry = _env_float("PAIR_Z_ENTRY", 2.0)
        self.pair_z_exit = _env_float("PAIR_Z_EXIT", 0.5)
        self.pair_z_stop = _env_float("PAIR_Z_STOP", 3.5)

        self.max_pair_pct = _env_float("MAX_PAIR_PCT", 0.03)
        self.max_pairs_pct = _env_float("MAX_PAIRS_PCT", 0.30)

        self.pair_book_path = os.environ.get("PAIR_BOOK_PATH", "data/pairs_active.json")

        self.pair_kelly_enabled = _env_bool("PAIR_KELLY_ENABLED", True)
        self.pair_kelly_fraction_cap = _env_float("PAIR_KELLY_FRACTION_CAP", 0.5)
        self.pair_kelly_max_mult = _env_float("PAIR_KELLY_MAX_MULT", 1.5)
        self.pair_backtest_stats_path = os.environ.get(
            "PAIR_BACKTEST_STATS_PATH", "data/pairs_backtest_stats.json"
        )

    #: Attribute names whose values are never rendered. Without this, the
    #: default object repr dumps every credential in cleartext -- and a repr
    #: reaches stdout, pytest assertion output, the orchestrator's JSONL log and
    #: the Reviewer's prompt, so `assert get_settings().alpaca_secret_key` (or
    #: any failing assertion that reprs the object) would publish the brokerage
    #: secret three ways.
    _SECRET_ATTRS = frozenset({
        "alpaca_api_key", "alpaca_secret_key", "openrouter_api_key",
        "edgar_identity", "finra_api_key", "securitiesdb_api_key",
    })

    def __repr__(self) -> str:
        parts = []
        for key, value in sorted(vars(self).items()):
            if key in self._SECRET_ATTRS:
                shown = "[REDACTED]" if value else "''"
            else:
                shown = repr(value)
            parts.append(f"{key}={shown}")
        return f"Settings({', '.join(parts)})"

    __str__ = __repr__


def get_settings() -> Settings:
    return Settings()

# Module-level singleton instance
SETTINGS = get_settings()


def reset_settings_cache() -> None:
    """Compatibility hook: clear any module-level settings cache.

    get_settings() constructs a fresh Settings() per call, so there is
    nothing to clear; this hook exists so cache-based implementations
    can be swapped in later without changing call sites.
    """
    return None
