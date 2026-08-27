"""Conservative static checks for LLM-generated descriptor functions.

Generated code is never executed by the compilation pipeline. These checks are
an audit aid, not a security sandbox; human review remains required.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass


FORBIDDEN_CALLS = {
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "__import__",
}

FORBIDDEN_ROOTS = {
    "builtins",
    "http",
    "MPRester",
    "mp_api",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "torch",
    "urllib",
    "urlopen",
}


@dataclass(frozen=True)
class AuditResult:
    passed: bool
    function_name: str | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.value if isinstance(node, ast.Attribute) else node.func
    return node.id if isinstance(node, ast.Name) else None


def audit_generated_code(code: str, expected_name: str) -> AuditResult:
    """Check syntax, interface, imports, and obvious external-access primitives."""
    issues: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return AuditResult(False, None, (f"syntax_error:{exc.msg}",))

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    non_functions = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
    ]
    function_name = functions[0].name if len(functions) == 1 else None
    if len(functions) != 1:
        issues.append(f"top_level_function_count:{len(functions)}")
    if non_functions:
        issues.append("top_level_statements_present")
    if function_name != expected_name:
        issues.append(f"unexpected_function_name:{function_name}")

    if functions:
        fn = functions[0]
        args = fn.args
        positional = [*args.posonlyargs, *args.args]
        if [arg.arg for arg in positional] != ["structure"]:
            issues.append("signature_must_be_structure_only")
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults:
            issues.append("optional_or_variadic_arguments_present")
        if isinstance(fn, ast.AsyncFunctionDef):
            issues.append("async_function_not_allowed")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            issues.append("import_statement_present")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                issues.append(f"forbidden_call:{node.func.id}")
            root = _root_name(node.func)
            if root in FORBIDDEN_ROOTS:
                issues.append(f"forbidden_access:{root}")

    unique = tuple(dict.fromkeys(issues))
    return AuditResult(not unique, function_name, unique)
