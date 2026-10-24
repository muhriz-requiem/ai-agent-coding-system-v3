"""Tool registry: JSON-schema tool signatures grouped by category.

Orchestrators resolve the *minimal* tool set per agent call
(:func:`resolve_required_tools`) and inject only that set into the prompt.
This keeps token usage low and limits what each role can do.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import config
from validation import TaskSpec

# ---------------------------------------------------------------------------
# Tool definitions (name -> descriptor with a JSON-schema ``parameters``)
# ---------------------------------------------------------------------------
def _tool(name: str, category: str, description: str,
          properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


TOOLS: Dict[str, Dict[str, Any]] = {
    # ---- fs ------------------------------------------------------------
    "fs.read_file": _tool(
        "fs.read_file", "fs", "Read a text file from the project workspace.",
        {"path": {"type": "string", "description": "Project-relative file path"}},
        ["path"],
    ),
    "fs.write_file": _tool(
        "fs.write_file", "fs", "Create or overwrite a file with new content.",
        {
            "path": {"type": "string", "description": "Project-relative file path"},
            "content": {"type": "string", "description": "Full file content"},
        },
        ["path", "content"],
    ),
    "fs.apply_patch": _tool(
        "fs.apply_patch", "fs", "Apply a unified diff to the project tree.",
        {"diff": {"type": "string", "description": "Unified diff text"}},
        ["diff"],
    ),
    # ---- test ------------------------------------------------------------
    "test.run_tests": _tool(
        "test.run_tests", "test", "Run the project test suite (pytest).",
        {"path": {"type": "string", "description": "Optional test path or node id"}},
    ),
    # ---- lint ------------------------------------------------------------
    "lint.run_lint": _tool(
        "lint.run_lint", "lint", "Run the configured linter over changed files.",
        {"files": {"type": "array", "items": {"type": "string"}}},
    ),
    "lint.type_check": _tool(
        "lint.type_check", "lint", "Run the static type checker (mypy-style).",
        {"files": {"type": "array", "items": {"type": "string"}}},
    ),
    # ---- git ------------------------------------------------------------
    "git.commit": _tool(
        "git.commit", "git", "Commit staged changes with a message.",
        {"message": {"type": "string"}},
        ["message"],
    ),
    # ---- search ------------------------------------------------------------
    "search.code_search": _tool(
        "search.code_search", "search",
        "Search project sources for a regex pattern.",
        {
            "query": {"type": "string"},
            "glob": {"type": "string", "description": "Optional file glob filter"},
        },
        ["query"],
    ),
    # ---- data / experiments (v2.6 objective gate + Kaggle pack) ----------
    "data.profile": _tool(
        "data.profile", "data",
        "Profile a dataset: schema, dtypes, missingness, target balance.",
        {
            "path": {"type": "string", "description": "Project-relative data path"},
            "target": {"type": "string", "description": "Optional target column"},
        },
        ["path"],
    ),
    "experiment.run": _tool(
        "experiment.run", "data",
        "Run a pinned experiment script; it must write the metrics JSON.",
        {
            "command": {"type": "string", "description": "Command to execute"},
            "results_path": {"type": "string",
                             "description": "Where metrics JSON is written"},
        },
        ["command"],
    ),
    "metrics.read": _tool(
        "metrics.read", "data",
        "Read the experiment/metrics JSON written by the last run.",
        {"path": {"type": "string"}},
    ),
    "benchmark.run": _tool(
        "benchmark.run", "test",
        "Run a performance benchmark producing a pinned metric.",
        {"command": {"type": "string"}, "results_path": {"type": "string"}},
        ["command"],
    ),
    # ---- submissions (Kaggle domain pack) ---------------------------------
    "submission.validate": _tool(
        "submission.validate", "submission",
        "Validate a submission file against the sample-submission format.",
        {
            "path": {"type": "string"},
            "sample": {"type": "string",
                       "description": "Reference sample submission file"},
        },
        ["path"],
    ),
    "submission.write": _tool(
        "submission.write", "submission",
        "Write the final submission file (id + prediction columns).",
        {"path": {"type": "string"}},
        ["path"],
    ),
    "kaggle.submit": _tool(
        "kaggle.submit", "submission",
        "Submit a validated file to the competition (CV-gated, capped).",
        {"file": {"type": "string"}, "message": {"type": "string"}},
        ["file"],
    ),
    "kaggle.competition_info": _tool(
        "kaggle.competition_info", "submission",
        "Read competition metadata: metric, files, submissions used.",
        {"competition": {"type": "string"}},
        ["competition"],
    ),
    # ---- research (v3 Phase 4: gated, prompt-injected lane) ----------------
    "research.analyze_app": _tool(
        "research.analyze_app", "research",
        "Sliced REA Evidence for one workspace-contained app artifact "
        "(loaded only when config.REA_RESEARCH_ENABLED is on).",
        {"app_path": {"type": "string",
                      "description": "Absolute path inside the workspace"}},
        ["app_path"],
    ),
    "research.prior_art": _tool(
        "research.prior_art", "research",
        "Digest of GitHub repos to learn ideas from (licenses pinned, "
        "auth-neutral; loaded only when config.PRIOR_ART_ENABLED is on).",
        {"topic": {"type": "string",
                   "description": "Library or technique to survey"}},
        ["topic"],
    ),
}

#: Minimal per-role tool sets (Builder: fs+lint; Reviewer: test+search;
#: Planner: search only -- orchestrators inject ONLY the resolved set).
ROLE_TOOL_MAP: Dict[str, List[str]] = {
    "builder": ["fs.read_file", "fs.write_file", "fs.apply_patch",
                "lint.run_lint", "lint.type_check"],
    "reviewer": ["test.run_tests", "search.code_search"],
    "planner": ["search.code_search"],
    # v2.6 roles (objective gate / divergence / Kaggle pack)
    "eda": ["data.profile", "search.code_search"],
    "feature_engineer": ["data.profile", "fs.read_file", "fs.write_file"],
    "modeler": ["experiment.run", "metrics.read", "fs.read_file",
                "fs.write_file"],
    "ensembler": ["experiment.run", "metrics.read"],
    "submission_validator": ["submission.validate", "submission.write"],
    "judge": ["test.run_tests", "search.code_search"],
    "merger": ["fs.read_file", "fs.write_file"],
    "strategist": ["search.code_search"],
    "red_team": ["test.run_tests", "search.code_search"],
    "campaign": ["data.profile", "search.code_search"],
    # NOTE: this key was previously listed twice.  A dict literal silently
    # keeps the last binding, so the director's ``metrics.read`` (needed to see
    # the scoreboard it is steering by) was unreachable.  Restored.
    "campaign_director": ["data.profile", "metrics.read",
                          "search.code_search"],
}


#: The two roles allowed to carry research-lane descriptors when their gates
#: are ON (planner briefs, builder implements -- see ``research.py``).
_RESEARCH_ROLES = ("builder", "planner")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def list_categories() -> List[str]:
    """Return the sorted list of tool categories (fs, test, lint, git, search)."""
    return sorted({tool["category"] for tool in TOOLS.values()})


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Return a tool descriptor by name, or ``None`` when unknown."""
    return TOOLS.get(name)


def resolve_required_tools(task: TaskSpec, role: str) -> List[Dict[str, Any]]:
    """Resolve the minimal tool descriptor set for *role* handling *task*.

    The task currently refines nothing (sets are already minimal), but the
    hook exists so future policies (e.g. git tools for bugfixes) can slot in
    without touching orchestrators.

    v3 research lane: the ``research.*`` descriptors attach only while their
    config gate is ON, and only for planner/builder.  With every gate at its
    default this returns exactly the pre-research-lane sets, so prompts do
    not shift when the lane is off.
    """
    role_key = (role or "").lower()
    names = list(ROLE_TOOL_MAP.get(role_key, []))
    if role_key in _RESEARCH_ROLES:
        if config.REA_RESEARCH_ENABLED and "research.analyze_app" in TOOLS:
            names.append("research.analyze_app")
        if config.PRIOR_ART_ENABLED and "research.prior_art" in TOOLS:
            names.append("research.prior_art")
    return [TOOLS[name] for name in names if name in TOOLS]
