"""Spec-author test paths disambiguate among plan siblings -- 2026-10-17.

Fix from the boardscan authoring attempt: three tasks all targeting
``src/boardscan.py`` all inherited the same fallback test path
``tests/test_boardscan.py`` from the module-stem rule, so ``write_tests``
(recorded contract: never overwrite) held tasks 2..N with "test_path already
exists" and they were silently lost at board fill. Derived (fallback) paths
now get title-suffixed when they collide; explicit paths are never rewritten.
"""

import spec_author


def _draft(task_id: str, title: str) -> dict:
    return {"id": task_id, "title": title, "objective": f"{title} work",
            "target_files": ["src/boardscan.py"]}


def test_same_module_drafts_get_distinct_fallback_paths():
    plan = spec_author.coerce_plan(
        {"tasks": [_draft("T-1", "Loader"), _draft("T-2", "Renderer"),
                   _draft("T-3", "CLI")]},
        goal="boardscan", existing_ids=[], require_tests=False)
    paths = [t.test_path for t in plan.usable]
    assert len(set(paths)) == len(paths), paths
    assert paths[0] == "tests/test_boardscan.py"  # first claimant unchanged
    assert "_renderer" in paths[1] and "_cli" in paths[2]


def test_explicit_path_is_never_rewritten_and_derived_moves_aside():
    first = _draft("T-1", "Loader")
    first["test_path"] = "tests/test_boardscan.py"  # model named it itself
    plan = spec_author.coerce_plan(
        {"tasks": [first, _draft("T-2", "Renderer")]},
        goal="boardscan", existing_ids=[], require_tests=False)
    paths = [t.test_path for t in plan.usable]
    assert paths[0] == "tests/test_boardscan.py"  # explicit stays exact
    assert paths[1] != paths[0], paths


def test_explicit_duplicate_keeps_existing_hard_stop_semantics():
    one = _draft("T-1", "Loader")
    one["test_path"] = "tests/test_contract.py"
    two = _draft("T-2", "Renderer")
    two["test_path"] = "tests/test_contract.py"
    plan = spec_author.coerce_plan(
        {"tasks": [one, two]},
        goal="g", existing_ids=[], require_tests=False)
    # coerce_plan must not intervene between what the model named explicitly
    # and write_tests' recorded refusal -- an explicit duplicate stays so.
    paths = [t.test_path for t in plan.usable]
    assert paths[0] == paths[1] == "tests/test_contract.py", paths


def test_duplicate_titles_fall_back_to_numerals():
    plan = spec_author.coerce_plan(
        {"tasks": [_draft("T-1", "Loader"), _draft("T-2", "Loader"),
                   _draft("T-3", "Loader")]},
        goal="g", existing_ids=[], require_tests=False)
    paths = [t.test_path for t in plan.usable]
    assert paths[0] == "tests/test_boardscan.py"
    assert paths[1] == "tests/test_boardscan_loader.py"
    assert paths[2] == "tests/test_boardscan_loader_2.py", paths


def test_distinct_modules_are_untouched():
    drafts = [_draft("T-1", "Loader"), _draft("T-2", "Cli")]
    drafts[0]["target_files"] = ["src/loader.py"]
    drafts[1]["target_files"] = ["src/cli.py"]
    plan = spec_author.coerce_plan(
        {"tasks": drafts}, goal="g", existing_ids=[], require_tests=False)
    paths = [t.test_path for t in plan.usable]
    assert paths == ["tests/test_loader.py", "tests/test_cli.py"], paths
