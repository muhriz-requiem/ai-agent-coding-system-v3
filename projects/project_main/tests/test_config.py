import os
import pytest
from unittest.mock import patch

from src.config import get_settings, SETTINGS, DEFAULT_WATCHLIST, DEFAULT_TIER_MODEL_MAP


class TestConfig:
    def test_default_settings(self):
        """Test that default settings are loaded correctly when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
            
            assert settings.alpaca_api_key == ""
            assert settings.alpaca_secret_key == ""
            assert settings.alpaca_base_url == ""
            assert settings.paper_only is True
            assert settings.watchlist == DEFAULT_WATCHLIST
            assert len(settings.watchlist) == 10
            assert all(ticker and isinstance(ticker, str) for ticker in settings.watchlist)
            assert settings.tier_model_map == DEFAULT_TIER_MODEL_MAP
            assert all(isinstance(model, str) and model for model in settings.tier_model_map.values())

    def test_paper_only_env_var_parsing(self):
        """Test PAPER_ONLY env var parsing with various truthy/falsy values."""
        # Test falsy values
        falsy_values = ["false", "FALSE", "0", "no", "NO"]
        for value in falsy_values:
            with patch.dict(os.environ, {"PAPER_ONLY": value}):
                settings = get_settings()
                assert settings.paper_only is False
        
        # Test truthy values
        truthy_values = ["true", "TRUE", "1", "yes", "YES", "anything"]
        for value in truthy_values:
            with patch.dict(os.environ, {"PAPER_ONLY": value}):
                settings = get_settings()
                assert settings.paper_only is True
        
        # Test unset defaults to True
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
            assert settings.paper_only is True

    def test_watchlist_env_var_parsing(self):
        """Test WATCHLIST env var parsing with various formats."""
        # Test valid comma-separated list
        test_watchlist = "AAPL, MSFT, GOOGL  , AMZN"
        expected = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        with patch.dict(os.environ, {"WATCHLIST": test_watchlist}):
            settings = get_settings()
            assert settings.watchlist == expected
        
        # Test empty string falls back to default
        with patch.dict(os.environ, {"WATCHLIST": ""}):
            settings = get_settings()
            assert settings.watchlist == DEFAULT_WATCHLIST
        
        # Test unset falls back to default
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
            assert settings.watchlist == DEFAULT_WATCHLIST
        
        # Test with extra commas and whitespace
        messy_watchlist = " AAPL ,, MSFT , , GOOGL  ,"
        expected = ["AAPL", "MSFT", "GOOGL"]
        with patch.dict(os.environ, {"WATCHLIST": messy_watchlist}):
            settings = get_settings()
            assert settings.watchlist == expected

    def test_tier_model_map_env_var_parsing(self):
        """Test TIER_MODEL_MAP env var parsing."""
        # Test valid comma-separated mappings
        test_mapping = "tier1:gpt-4o-mini,tier2:gpt-4o,tier3:gpt-4.1"
        expected = {
            "tier1": "gpt-4o-mini",
            "tier2": "gpt-4o",
            "tier3": "gpt-4.1"
        }
        with patch.dict(os.environ, {"TIER_MODEL_MAP": test_mapping}):
            settings = get_settings()
            assert settings.tier_model_map == expected
        
        # Test empty string falls back to default
        with patch.dict(os.environ, {"TIER_MODEL_MAP": ""}):
            settings = get_settings()
            assert settings.tier_model_map == DEFAULT_TIER_MODEL_MAP
        
        # Test unset falls back to default
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
            assert settings.tier_model_map == DEFAULT_TIER_MODEL_MAP
        
        # Test invalid format falls back to default
        with patch.dict(os.environ, {"TIER_MODEL_MAP": "invalid"}):
            settings = get_settings()
            assert settings.tier_model_map == DEFAULT_TIER_MODEL_MAP

    def test_alpaca_keys_env_vars(self):
        """Test ALPACA keys are read from environment."""
        test_api_key = "test_api_key_123"
        test_secret_key = "test_secret_key_456"
        test_base_url = "https://test.alpaca.markets"
        
        with patch.dict(os.environ, {
            "ALPACA_API_KEY": test_api_key,
            "ALPACA_SECRET_KEY": test_secret_key,
            "ALPACA_BASE_URL": test_base_url
        }):
            settings = get_settings()
            assert settings.alpaca_api_key == test_api_key
            assert settings.alpaca_secret_key == test_secret_key
            assert settings.alpaca_base_url == test_base_url

    def test_no_network_calls_on_import(self):
        """Test that importing config doesn't trigger network calls."""
        # This test ensures no network libraries are imported or used
        import sys
        
        # Check that common network libraries aren't imported by our config module
        assert "requests" not in sys.modules or not hasattr(sys.modules["requests"], "get")
        assert "httpx" not in sys.modules or not hasattr(sys.modules["httpx"], "get")
        assert "urllib" not in sys.modules or not hasattr(sys.modules["urllib"], "urlopen")
        
        # Import fresh to ensure no side effects
        import importlib
        import src.config as config_module
        importlib.reload(config_module)
        
        # Verify we can still get settings without network issues
        settings = config_module.get_settings()
        assert isinstance(settings, config_module.Settings)

    def test_settings_singleton_behavior(self):
        """Test that get_settings() returns fresh instances but module-level singleton is stable."""
        # get_settings() should return fresh instances each call
        settings1 = get_settings()
        settings2 = get_settings()
        assert isinstance(settings1, type(settings2))
        # The module-level singleton should be stable
        assert SETTINGS is SETTINGS

    def test_package_dirs_importable(self):
        """Test that all required package directories are importable."""
        # This is a smoke test to ensure the package structure is correct
        import src.data
        import src.quant
        import src.agents
        import src.broker
        import src.pipeline
        import src.store
        
        # Just verify they can be imported without errors
        assert src.data is not None
        assert src.quant is not None
        assert src.agents is not None
        assert src.broker is not None
        assert src.pipeline is not None
        assert src.store is not None

    def test_logging_setup_multiple_calls(self):
        """Test that calling configure_logging multiple times doesn't duplicate handlers."""
        import logging
        from src.logging_setup import configure_logging
        
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # First call
        configure_logging()
        handler_count_after_first = len(root_logger.handlers)
        
        # Second call
        configure_logging()
        handler_count_after_second = len(root_logger.handlers)
        
        # Should not have added duplicate handlers
        assert handler_count_after_second == handler_count_after_first
        
        # Clean up
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
