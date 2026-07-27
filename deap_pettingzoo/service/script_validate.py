"""Static validation for user-submitted experiment scripts.

Backs ``POST /experiments/script/validate``. This is **static** analysis only:
it confirms the script parses and looks like a runnable experiment, but it
cannot prove the script trains correctly — only actually running it can. The UI
states this caveat next to the Validate button.

Checks:

* ``ast.parse`` — a ``SyntaxError`` becomes a blocking error with its line number.
* Structure — a warning if neither ``if __name__ == "__main__"`` nor a top-level
  ``main()`` is present (the script may not do anything when executed).
* Imports — a warning if the script imports ``examples.*`` modules, which are
  only importable in the backend when ``examples/`` is on ``PYTHONPATH`` (true in
  the provided container image, but worth surfacing).
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.If):
            test = node.test
            # if __name__ == "__main__":
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                return True
    return False


def _has_main_func(tree: ast.Module) -> bool:
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"
        for n in tree.body
    )


def _imports_examples(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "examples":
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "examples":
                    return True
    return False


def validate_script_source(src: str) -> Dict[str, Any]:
    """Return a ``{"ok": bool, "issues": [...]}`` report (ValidationReport shape)."""
    issues: List[Dict[str, str]] = []

    if not src or not src.strip():
        return {
            "ok": False,
            "issues": [
                {
                    "path": "script",
                    "code": "empty",
                    "message": "Script is empty.",
                    "severity": "error",
                }
            ],
        }

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        loc = f" (line {exc.lineno})" if exc.lineno else ""
        return {
            "ok": False,
            "issues": [
                {
                    "path": "script",
                    "code": "syntax_error",
                    "message": f"{exc.msg}{loc}",
                    "severity": "error",
                }
            ],
        }

    if not _has_main_guard(tree) and not _has_main_func(tree):
        issues.append(
            {
                "path": "script",
                "code": "no_entrypoint",
                "message": (
                    'No `if __name__ == "__main__":` block or top-level `main()` '
                    "found — the script may not run anything when executed."
                ),
                "severity": "warning",
            }
        )

    if _imports_examples(tree):
        issues.append(
            {
                "path": "script",
                "code": "examples_import",
                "message": (
                    "Script imports `examples.*` modules. These resolve only when "
                    "the backend has `examples/` on PYTHONPATH (true in the provided "
                    "container image)."
                ),
                "severity": "warning",
            }
        )

    # Warnings/info never block submission.
    return {"ok": True, "issues": issues}
