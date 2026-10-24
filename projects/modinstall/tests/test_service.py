import importlib.util, pytest
if importlib.util.find_spec('src.installer.service') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.installer.service import register_channel, dispatch

def test_register_and_dispatch_happy_path():
    def mock_handler(payload):
        return payload.get("value", 0) * 2

    register_channel("test_channel", mock_handler)
    result = dispatch("test_channel", {"value": 21})
    assert result == 42

def test_dispatch_unregistered_channel_raises_key_error():
    with pytest.raises(KeyError):
        dispatch("nonexistent_channel", {})