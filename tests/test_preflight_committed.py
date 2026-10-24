"""``--preflight scan --committed``: the check that can actually go green.

``--preflight scan`` answers "is there credential-shaped content anywhere in this
working tree".  That is the right question for a *directory* and the wrong one
for a *repository*, and it can never pass: this repo carries a live ``.env`` (a
secret), a historical ``agent_log.jsonl`` and a ``review_history.log`` (redacted
but still matching), and five scanner test fixtures whose entire purpose is to
contain an AWS key id, a GitHub token and a PEM header.  63 findings, forever.

A check that is always red is a check nobody reads -- and that is how a real
leak ships: the operator learns to skim the output, so the day a genuine
credential appears among the noise, it looks like the usual noise.

So ``--committed`` asks the question that has a green answer: *could a secret
reach a commit?*  That requires three properties, each of which is a hole if it
is missing:

1. it must scan exactly what git tracks (not the tree, not a hand-rolled
   ``.gitignore`` reimplementation that can drift);
2. it must have a way to exempt a file that *must* contain secret-shaped text,
   and that exemption must never be silent;
3. it must still have teeth -- a real secret in a non-exempt tracked file has to
   make it red again.

These tests pin all three, plus the scope boundaries.
"""

from __future__ import annotations

import subprocess

import pytest

import main
import redaction


# --------------------------------------------------------------- the allowlist
def test_allowlist_is_empty_when_absent(tmp_path):
    assert redaction.scan_allowlist(tmp_path) == []


def test_allowlist_reads_paths_and_ignores_comments(tmp_path):
    (tmp_path / ".secret-scan-allow").write_text(
        "# a comment\n"
        "tests/test_redaction.py\n"
        "   tests/test_*_gates.py   \n"
        "\n"
        "tests/test_x.py  # trailing comment\n",
        encoding="utf-8")
    pats = redaction.scan_allowlist(tmp_path)
    assert pats == ["tests/test_redaction.py", "tests/test_*_gates.py",
                    "tests/test_x.py"]


def test_allowlist_matches_exact_and_glob(tmp_path):
    pats = ["tests/test_redaction.py", "tests/fixtures/*"]
    assert redaction.is_scan_allowed("tests/test_redaction.py", pats)
    assert redaction.is_scan_allowed("tests/fixtures/keys.txt", pats)
    assert not redaction.is_scan_allowed("tests/test_other.py", pats)
    assert not redaction.is_scan_allowed("src/app.py", pats)
    assert not redaction.is_scan_allowed("anything", [])


def test_allowlist_normalises_windows_separators(tmp_path):
    (tmp_path / ".secret-scan-allow").write_text(
        "tests\\test_redaction.py\n", encoding="utf-8")
    assert redaction.is_scan_allowed("tests/test_redaction.py",
                                     redaction.scan_allowlist(tmp_path))


# ------------------------------------------------------------ tracked file set
def test_tracked_files_refuses_to_guess_when_git_is_unusable(tmp_path):
    """The critical property: no silent degradation.

    Returning "everything" as a fallback would produce a green result for a
    question the caller never actually got answered.
    """
    paths, note = redaction.tracked_files(tmp_path)
    assert paths == []
    assert note, "a refusal must always explain itself"


def test_tracked_files_lists_only_tracked_files(tmp_path):
    def git(*args, cwd=tmp_path):
        return subprocess.run(["git", "-C", str(cwd), *args],
                              capture_output=True, check=False)

    if git("init").returncode != 0:
        pytest.skip("git unavailable")
    (tmp_path / "tracked.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("hello\n", encoding="utf-8")
    git("add", "tracked.txt", ".gitignore")

    paths, note = redaction.tracked_files(tmp_path)
    names = sorted(p.name for p in paths)
    assert names == [".gitignore", "tracked.txt"], names
    assert "tracked" in note


# ------------------------------------------------------------- scan_paths basics
def test_scan_paths_scans_the_explicit_list(tmp_path):
    good = tmp_path / "clean.py"
    good.write_text("x = 1\n", encoding="utf-8")
    bad = tmp_path / "leak.py"
    bad.write_text('AWS = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
    missing = tmp_path / "gone.py"

    findings = redaction.scan_paths([good, bad, missing])
    assert len(findings) == 1
    assert findings[0].get("path") == str(bad)


def test_scan_paths_ignores_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    assert redaction.scan_paths([tmp_path / "sub", tmp_path]) == []


def test_scan_paths_of_nothing_is_empty(tmp_path):
    assert redaction.scan_paths([]) == []
    assert redaction.scan_paths(None) == []


# --------------------------------------------------- the teeth of the whole mode
def test_a_real_secret_in_a_tracked_file_makes_it_red(tmp_path, capsys):
    """The property that makes the exemption mechanism safe.

    The allowlist exempts five files. If it exempted *everything*, or if the
    scope silently widened to the whole tree, this check would be theatre.
    """
    import contextlib
    import io

    def run(ws):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main._preflight_command("scan", ws, committed=True)
        return code, buf.getvalue()

    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args],
                              capture_output=True, check=False)

    if git("init").returncode != 0:
        pytest.skip("git unavailable")

    fixture = tmp_path / "tests" / "test_redaction.py"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text('AWS = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
    (tmp_path / ".secret-scan-allow").write_text(
        "# the scanner fixtures\ntests/test_redaction.py\n", encoding="utf-8")
    git("add", "tests/test_redaction.py", ".secret-scan-allow")

    # Exempt: green, and the exemption is disclosed.
    code, out = run(tmp_path)
    assert code == 0, out
    assert "clean" in out
    assert ".secret-scan-allow" in out
    assert "tests/test_redaction.py" in out, (
        "an exemption must never be silent -- the operator has to see it")

    # Not exempt: red again. This is the assertion that matters.
    leaky = tmp_path / "src" / "config.py"
    leaky.parent.mkdir(parents=True, exist_ok=True)
    leaky.write_text(
        'GITHUB_TOKEN = "ghp_16CharactersOfNonsense1234567890abcdef"\n',
        encoding="utf-8")
    git("add", "src/config.py")
    code, out = run(tmp_path)
    assert code == 1, out
    assert "github_token" in out
    assert "src/config.py" in out or "config.py" in out


def test_whole_tree_scan_ignores_the_allowlist(tmp_path):
    """Suppressing a committed-file finding must not silence the wider sweep."""
    leaky = tmp_path / "tests" / "test_redaction.py"
    leaky.parent.mkdir(parents=True)
    leaky.write_text('AWS = "AKIAIOSFODNN7EXAMPLE"\n', encoding="utf-8")
    (tmp_path / ".secret-scan-allow").write_text(
        "tests/test_redaction.py\n", encoding="utf-8")

    findings = redaction.scan_tree(tmp_path)
    assert any("test_redaction.py" in str(f.get("path")) for f in findings), (
        "the whole-tree scan must still report an exempt file")


def test_committed_flag_is_refused_outside_scan():
    """A flag that silently applies to the wrong verb is a trap."""
    for argv in (["--committed"],
                 ["--preflight", "scrub", "--committed"],
                 ["--preflight", "paths", "--committed"]):
        assert main.main(argv) == 2, argv


def test_apply_flag_is_still_scoped_to_scrub():
    assert main.main(["--preflight", "scan", "--apply"]) == 2