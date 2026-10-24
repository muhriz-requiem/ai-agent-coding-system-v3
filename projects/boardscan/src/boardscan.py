import argparse
import json
import os
import re
import sys

def load_projects(root: str, project: str | None = None) -> list[tuple[str, object]]:
    """
    Load projects from <root>/projects/* (directories only, sorted ascending by name).
    - If root is not an existing directory, raise SystemExit(2).
    - If project is given and <root>/projects/<project> is not a directory, 
      print 'no such project: <project>' to stderr and raise SystemExit(2).
    - Skip a project entirely when its .agent/task_board.json is missing.
    - Return (name, parsed-board-dict) for readable boards and (name, exception) 
      for unreadable ones (JSONDecodeError/OSError).
    """
    root_path = os.path.abspath(root) if root else ""
    if not root or not os.path.isdir(root_path):
        raise SystemExit(2)

    projects_dir = os.path.join(root_path, "projects")

    if project is not None:
        proj_path = os.path.join(projects_dir, project)
        if not os.path.isdir(proj_path):
            sys.stderr.write(f"no such project: {project}\n")
            raise SystemExit(2)

    if not os.path.isdir(projects_dir):
        return []

    try:
        entries = sorted(
            [e.name for e in os.scandir(projects_dir) if e.is_dir()],
            key=lambda s: s
        )
    except OSError:
        return []

    if project is not None:
        if project in entries:
            entries = [project]
        else:
            entries = []

    results: list[tuple[str, object]] = []

    for name in entries:
        board_path = os.path.join(projects_dir, name, ".agent", "task_board.json")
        if not os.path.isfile(board_path):
            continue

        try:
            with open(board_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append((name, data))
        except (json.JSONDecodeError, OSError) as e:
            results.append((name, e))

    return results

def _normalize_text(text: str) -> str:
    if not text:
        return ""
    # Replace \r\n first with single space, then remaining \n and \r with single space
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text

def format_task_row(task: dict) -> str:
    """
    Render one task as columns [id, TITLE, STATUS, ATTEMPTS, DEPS] joined by ' | '.
    - id and title: replace every '\r\n' with a single space, then remaining '\n' and '\r' with a single space.
    - STATUS: verbatim when one of done|pending|failed else '?'
    - ATTEMPTS: str(task['retry_count']) when key exists, else '0'
    - DEPS: ', '.join(dependencies) when dependencies is a non-empty list, else '-'
    """
    raw_id = str(task.get("id", ""))
    raw_title = str(task.get("title", ""))

    tid = _normalize_text(raw_id)
    ttitle = _normalize_text(raw_title)

    status = task.get("status")
    if status not in ("done", "pending", "failed"):
        status = "?"

    attempts = str(task["retry_count"]) if "retry_count" in task else "0"

    deps = task.get("dependencies")
    if isinstance(deps, list) and deps:
        deps_str = ", ".join(str(d) for d in deps)
    else:
        deps_str = "-"

    return f"{tid} | {ttitle} | {status} | {attempts} | {deps_str}"

def parse_diagnosis(feedback: str | None) -> dict | None:
    """
    Only feedback that starts with 'DIAGNOSIS:' AND matches
    'DIAGNOSIS: <cause> (confidence=<confidence>) remedy=<remedy>' returns
    {'cause': str, 'confidence': str, 'remedy': str}; otherwise None.
    """
    if not isinstance(feedback, str):
        return None
    
    pattern = r"^DIAGNOSIS: (.+) \(confidence=(.+)\) remedy=(.+)$"
    match = re.match(pattern, feedback)
    if not match:
        return None

    cause, confidence, remedy = match.groups()
    return {
        "cause": cause,
        "confidence": confidence,
        "remedy": remedy
    }

def render_table(projects: list) -> str:
    """
    Consuming load_projects' output (list of (name, board-or-exception) tuples):
    - Skip (name, exception) tuples silently.
    - For each readable board emit header '== <name> ==',
      one format_task_row line per task in board['tasks'] order (board without tasks renders header only),
      then one indented line '  DIAGNOSIS: <cause> (confidence=<confidence>) remedy=<remedy>' per task
      whose 'last_feedback' parses via parse_diagnosis.
    - Join all lines with '\\n' and no trailing newline.
    - Empty input returns ''.
    """
    lines: list[str] = []

    for item in projects:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        name, board = item
        if isinstance(board, BaseException) or not isinstance(board, dict):
            continue

        lines.append(f"== {name} ==")

        tasks = board.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    lines.append(format_task_row(task))
                    feedback = task.get("last_feedback")
                    parsed = parse_diagnosis(feedback)
                    if parsed is not None:
                        diag_line = f"  DIAGNOSIS: {parsed['cause']} (confidence={parsed['confidence']}) remedy={parsed['remedy']}"
                        lines.append(diag_line)

    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan and report on boardscan projects.")
    parser.add_argument("--root", help="Root directory containing projects")
    parser.add_argument("--project", help="Specific project name")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args(argv)

    root = args.root if args.root is not None else "."
    project = args.project
    as_json = args.json

    projects = load_projects(root, project)

    readable_projects = []
    unreadable_names = []

    for name, board in projects:
        if isinstance(board, BaseException):
            unreadable_names.append(name)
        else:
            readable_projects.append((name, board))

    if as_json:
        payload = []
        for name, board in readable_projects:
            tasks_list = []
            diagnoses_list = []
            tasks = board.get("tasks")
            if isinstance(tasks, list):
                for task in tasks:
                    if isinstance(task, dict):
                        tid = task.get("id", "")
                        title = task.get("title", "")
                        status = task.get("status")
                        retry_count = task.get("retry_count", 0)
                        deps = task.get("dependencies", [])

                        tasks_list.append({
                            "id": tid,
                            "title": title,
                            "status": status,
                            "retry_count": retry_count,
                            "dependencies": deps
                        })

                        feedback = task.get("last_feedback")
                        parsed = parse_diagnosis(feedback)
                        if parsed is not None:
                            diagnoses_list.append({
                                "task_id": tid,
                                "cause": parsed["cause"],
                                "confidence": parsed["confidence"],
                                "remedy": parsed["remedy"],
                                "raw": feedback
                            })

            payload.append({
                "project": name,
                "tasks": tasks_list,
                "diagnoses": diagnoses_list
            })

        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        table_output = render_table(projects)
        if table_output:
            sys.stdout.write(table_output + "\n")

    for unreadable in unreadable_names:
        sys.stderr.write(f"{unreadable}\n")

    if unreadable_names:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
