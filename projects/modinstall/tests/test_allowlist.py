import importlib.util, pytest
if importlib.util.find_spec('src.installer.allowlist') is None:
    pytest.skip('implementation not written yet', allow_module_level=True)
from src.installer.allowlist import check_url_allowed

def test_check_url_allowed_exact_match():
    assert check_url_allowed("https://example.com/path", ["example.com"]) is True

def test_check_url_allowed_subdomain():
    assert check_url_allowed("https://sub.example.com/path", ["example.com"]) is True

def test_check_url_allowed_not_allowed():
    assert check_url_allowed("https://malicious.com", ["example.com"]) is False

def test_check_url_allowed_partial_domain_mismatch():
    assert check_url_allowed("https://notexample.com", ["example.com"]) is False

def test_check_url_allowed_multiple_allowed_domains():
    assert check_url_allowed("https://foo.bar.org", ["example.com", "bar.org"]) is True

def test_check_url_allowed_malformed_url():
    with pytest.raises(ValueError):
        check_url_allowed("://malformed-url", ["example.com"])