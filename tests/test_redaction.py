"""Secret redaction primitives: exact-value redaction, detection, scrubbing."""

from __future__ import annotations

import json

import pytest

import config
import redaction

# A synthetic value that is obviously not a real credential but satisfies every
# shape rule (>= SECRET_SCAN_MIN_LEN, high character diversity, no placeholder
# token).  Tests must never depend on the operator's actual keys.
FAKE = "Zz9qW3eR7tYuIoP1aSdF4gH6jKlZxCvB2nM5"
FAKE_2 = "Qw7ErTyUiOpAsDfGhJkLzXcVbNm4Qw7ErTyU"


@pytest.fixture(autouse=True)
def _clean_secret_state(monkeypatch):
    """Isolate the discovered-secret cache around every test."""
    monkeypatch.setenv("REDACTION_TEST_API_KEY", FAKE)
    monkeypatch.setenv("REDACTION_TEST_SECRET", FAKE_2)
    monkeypatch.delenv("SECRET_SCAN_MIN_LEN", raising=False)
    redaction.reset_cache()
    yield
    redaction.reset_cache()


@pytest.fixture
def iso(tmp_path):
    """An isolated workspace root so no real .env is ever read."""
    redaction.reset_cache()
    return tmp_path


# ---------------------------------------------------------------------------
# known_secret_values / discovery
# ---------------------------------------------------------------------------
def test_discovers_credential_shaped_env_vars(iso):
    values = redaction.known_secret_values(iso, refresh=True)
    assert FAKE in values and FAKE_2 in values


def test_ignores_non_credential_env_names(monkeypatch, iso):
    # A path, a ticker list and a flag must never enter the redaction set: they
    # are ordinary configuration and blanking them would corrupt real output.
    monkeypatch.setenv("DECISIONS_DB", "data/decisions.sqlite3")
    monkeypatch.setenv("WATCHLIST", "AAPL,MSFT,NVDA,AMD,GOOG,AMZN,META,TSLA")
    monkeypatch.setenv("CACHE_ENABLED", "true")
    values = redaction.known_secret_values(iso, refresh=True)
    assert "data/decisions.sqlite3" not in values
    assert "AAPL,MSFT,NVDA,AMD,GOOG,AMZN,META,TSLA" not in values
    assert "true" not in values


def test_reads_project_env_files(iso):
    project = iso / "projects" / "proj"
    project.mkdir(parents=True)
    (project / ".env").write_text(
        f"PROJECT_ONLY_TOKEN={FAKE_2}\n# comment\nNOT_A_SECRET=hello\n",
        encoding="utf-8")
    values = redaction.known_secret_values(iso, refresh=True)
    assert FAKE_2 in values
    assert "hello" not in values


def test_placeholder_values_excluded(monkeypatch, iso):
    # A placeholder in one source must not stop the live value in another
    # source from being discovered (regression: first-wins shadowing).
    monkeypatch.setenv("SHADOWED_API_KEY", "YOUR_API_KEY_HERE")
    (iso / ".env").write_text(f"SHADOWED_API_KEY={FAKE}\n", encoding="utf-8")
    values = redaction.known_secret_values(iso, refresh=True)
    assert FAKE in values
    assert "YOUR_API_KEY_HERE" not in values


@pytest.mark.parametrize("placeholder", [
    "YOUR_API_KEY_HERE", "changeme", "xxxxxxxxxxxxxxxxxxxx", "placeholder",
    "insert-token-here", "0000000000000000",
])
def test_placeholder_shapes_rejected(placeholder):
    assert not redaction._usable_secret(placeholder, 8)


@pytest.mark.parametrize("real", [
    # A Cohere-shaped key containing "here" as a substring by coincidence --
    # the case that a substring-based placeholder filter silently fails, since
    # it would classify the credential as documentation and skip redaction.
    "AQ.Ab8RN6IepfskvzherestrngwxyzS5L0ZRggh6fRTnRJ9G9sw4XKZK9Shw",
    "sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r",
    "hunter2-hunter2-hunter2",
])
def test_realistic_secrets_accepted(real):
    assert redaction._usable_secret(real, 8)


# ---------------------------------------------------------------------------
# redact_known_secrets: the always-on primitive
# ---------------------------------------------------------------------------
def test_redacts_value_but_preserves_identifier(iso):
    text = f'ALPACA_API_KEY = "{FAKE}"  # os.getenv("ALPACA_API_KEY")'
    out = redaction.redact_known_secrets(text, iso)
    assert FAKE not in out
    assert redaction.REDACTION_MARK in out
    # The identifier and the lookup expression must survive intact, or the
    # Reviewer becomes blind on exactly the credential-handling code.
    assert 'os.getenv("ALPACA_API_KEY")' in out
    assert "ALPACA_API_KEY =" in out


def test_clean_text_is_returned_unchanged(iso):
    text = "def handler(request):\n    return {'ok': True}\n"
    assert redaction.redact_known_secrets(text, iso) is text


def test_non_string_input_is_safe(iso):
    assert redaction.redact_known_secrets(None, iso) is None
    assert redaction.redact_known_secrets(42, iso) == 42
    assert redaction.redact_known_secrets("", iso) == ""


def test_disabled_redaction_is_a_noop(iso, monkeypatch):
    monkeypatch.setattr(config, "REDACTION_ENABLED", False)
    redaction.reset_cache()
    text = f"key={FAKE}"
    assert redaction.redact_known_secrets(text, iso) == text


def test_redact_obj_walks_nested_structures(iso):
    payload = {"a": f"token {FAKE}", "b": [1, {"c": FAKE_2}], "d": 7}
    out = redaction.redact_obj(payload, iso)
    assert FAKE not in json.dumps(out)
    assert FAKE_2 not in json.dumps(out)
    assert out["d"] == 7


# ---------------------------------------------------------------------------
# scan_secrets: detection, never mutation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule,text", [
    ("openrouter_key", "k=sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r"),
    ("anthropic_key", "k=sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"),
    ("google_api_key", "k=AIzaSyD-abcdefghijklmnopqrstuvwxyz0123456"),
    ("aws_access_key_id", "k=AKIAIOSFODNN7EXAMPLE"),
    ("github_token", "k=ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ("slack_token", "k=xoxb-1234567890-abcdefghijkl"),
    ("private_key_block", "-----BEGIN RSA PRIVATE KEY-----"),
    ("openrouter_account_id", "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ"),
    ("openrouter_org_id", "org_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ"),
    ("generic_assignment", 'api_key = "A1b2C3d4E5f6G7h8I9j0"'),
])
def test_detects_each_rule(rule, text):
    assert rule in {f["rule"] for f in redaction.scan_secrets(text)}


@pytest.mark.parametrize("text", [
    'api_key = os.getenv("SOME_KEY")',
    'password = ""',
    'key = "identifier"',
    "MONKEY_APPLE = 3",
    "self.token_count += 1",
    "def tokenize(line): return line.split()",
    "The secret sauce is idempotency.",
    "user_id = 42",
])
def test_no_false_positives_on_ordinary_code(text):
    assert redaction.scan_secrets(text) == []


def test_scan_detects_known_value(iso):
    findings = redaction.scan_secrets(f"leaked {FAKE} here")
    assert "known_value" in {f["rule"] for f in findings}


def test_finding_never_contains_the_value(iso):
    text = f"k=sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r and {FAKE}"
    for finding in redaction.scan_secrets(text):
        blob = json.dumps(finding)
        assert FAKE not in blob
        assert "sk-or-v1-1a2b" not in blob
        assert len(finding["hint"]) == 8


def test_same_value_yields_same_hint(iso):
    text = f"{FAKE} ... {FAKE}"
    hints = {f["hint"] for f in redaction.scan_secrets(text)
             if f["rule"] == "known_value"}
    assert len(hints) == 1


def test_scan_does_not_mutate(iso):
    text = f"sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r {FAKE}"
    before = text
    redaction.scan_secrets(text)
    assert text == before


@pytest.mark.parametrize("rule,text", [
    # Regression class: a trailing \b after a counted character class stops
    # matching as soon as the real token carries more characters than the
    # nominal count.  Each of these is the nominal form plus extra characters.
    ("google_api_key", "k=AIza" + "a" * 44),
    ("openai_key", "k=sk-" + "b" * 51),
    ("aws_access_key_id", "k=AKIAIOSFODNN7EXAMPLEZZ"),
    ("openrouter_account_id", "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJEXTRA"),
    ("openrouter_org_id", "org_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJEXTRA"),
    ("anthropic_key", "sk-ant-api03-" + "c" * 30),
])
def test_detects_overlong_tokens(rule, text):
    assert rule in {f["rule"] for f in redaction.scan_secrets(text)}


# ---------------------------------------------------------------------------
# scrub_secrets: applying counterpart
# ---------------------------------------------------------------------------
def test_scrub_replaces_every_occurrence_without_duplicating(iso):
    text = ("user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ and "
            "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ end")
    out, found = redaction.scrub_secrets(text)
    assert "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ" not in out
    assert len(found) == 2
    # Regression: the replacement callback must return only the matched span.
    assert out.count("and") == 1
    assert len(out) < len(text)


def test_scrub_preserves_assignment_prefix(iso):
    out, _ = redaction.scrub_secrets('api_key = "A1b2C3d4E5f6G7h8I9j0"')
    assert out == 'api_key = "[REDACTED]"'


def test_scrub_handles_known_values(iso):
    out, found = redaction.scrub_secrets(f"leak {FAKE} leak {FAKE_2}")
    assert FAKE not in out and FAKE_2 not in out
    assert len(found) == 2


def test_scrub_is_a_noop_on_clean_text(iso):
    text = "def add(a, b): return a + b"
    out, found = redaction.scrub_secrets(text)
    assert out == text and found == []


# ---------------------------------------------------------------------------
# JSON scrub: dry run vs apply
# ---------------------------------------------------------------------------
def test_scrub_json_file_is_dry_run(iso):
    target = iso / "anti.json"
    original = json.dumps({"desc": "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ"})
    target.write_text(original, encoding="utf-8")
    count, findings = redaction.scrub_json_file(target)
    assert count == 1 and len(findings) == 1
    # Nothing written.
    assert target.read_text(encoding="utf-8") == original
    assert list(iso.glob("*.bak-*")) == []


def test_apply_scrub_backs_up_then_rewrites(iso):
    target = iso / "anti.json"
    target.write_text(json.dumps(
        {"records": [{"description": "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ"}]}),
        encoding="utf-8")
    count, backup, findings = redaction.apply_scrub_json_file(target)
    assert count == 1 and backup is not None and backup.exists()
    # The backup holds the original bytes, for rollback.
    assert "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ" in backup.read_text(encoding="utf-8")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ" not in json.dumps(payload)
    # Structure preserved.
    assert "records" in payload and "description" in payload["records"][0]


def test_apply_scrub_is_idempotent(iso):
    target = iso / "anti.json"
    target.write_text(json.dumps({"d": "user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ"}),
                      encoding="utf-8")
    first, _, _ = redaction.apply_scrub_json_file(target)
    second, backup, _ = redaction.apply_scrub_json_file(target)
    assert first == 1
    assert second == 0 and backup is None


def test_apply_scrub_skips_clean_and_broken_files(iso):
    clean = iso / "clean.json"
    clean.write_text(json.dumps({"d": "nothing to see"}), encoding="utf-8")
    assert redaction.apply_scrub_json_file(clean) == (0, None, [])

    broken = iso / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert redaction.apply_scrub_json_file(broken) == (0, None, [])
    assert broken.read_text(encoding="utf-8") == "{not json"


def test_scrub_handles_secret_in_json_key(iso):
    target = iso / "k.json"
    target.write_text(json.dumps({"user_3IQTf0xMV5Q7wOYkjAoPYoJSmQeJ": 1}),
                      encoding="utf-8")
    count, _ = redaction.scrub_json_file(target)
    assert count == 1


# ---------------------------------------------------------------------------
# scan_path / scan_tree
# ---------------------------------------------------------------------------
def test_scan_path_reports_and_tags_path(iso):
    target = iso / "creds.py"
    target.write_text("KEY = 'sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r'\n",
                      encoding="utf-8")
    findings = redaction.scan_path(target)
    assert findings and findings[0]["path"] == str(target)


def test_scan_path_tolerates_missing_and_binary(iso):
    assert redaction.scan_path(iso / "nope.txt") == []
    blob = iso / "blob.bin"
    blob.write_bytes(bytes(range(256)))
    assert redaction.scan_path(blob) == []


def test_scan_tree_skips_caches(iso):
    (iso / "__pycache__").mkdir()
    (iso / "__pycache__" / "x.pyc").write_text(
        "sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r", encoding="utf-8")
    (iso / "real.py").write_text(
        "sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r", encoding="utf-8")
    findings = redaction.scan_tree(iso)
    assert len(findings) == 1
    assert findings[0]["path"].endswith("real.py")