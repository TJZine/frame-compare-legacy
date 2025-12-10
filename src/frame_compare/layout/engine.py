"""Layout expression engine and evaluation utilities."""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

_SAFE_FUNCTION_TOKENS = {"abs", "min", "max"}

_ALLOWED_BOOL_OPS: Tuple[type[ast.boolop], ...] = (ast.And, ast.Or)
_ALLOWED_BIN_OPS: Tuple[type[ast.operator], ...] = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.FloorDiv,
)
_ALLOWED_UNARY_OPS: Tuple[type[ast.unaryop], ...] = (ast.Not, ast.UAdd, ast.USub)
_ALLOWED_COMPARE_OPS: Tuple[type[ast.cmpop], ...] = (
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

# Expression security limits to prevent DoS via resource exhaustion
_MAX_AST_NODES = 50
_MAX_NUMERIC_CONSTANT = 10**9
_MAX_STRING_LENGTH = 1000


def _count_nodes(node: ast.AST) -> int:
    """Count total AST nodes for complexity budget."""
    return 1 + sum(_count_nodes(child) for child in ast.iter_child_nodes(node))


def validate_safe_expression(
    node: ast.AST, *, allowed_calls: Mapping[str, Any], allowed_names: Mapping[str, Any]
) -> None:
    """Ensure the parsed AST only contains whitelisted operations and bounded values."""
    node_count = _count_nodes(node)
    if node_count > _MAX_AST_NODES:
        raise ValueError(f"Expression too complex: {node_count} nodes exceeds limit of {_MAX_AST_NODES}")

    def _check(inner: ast.AST) -> None:
        if isinstance(inner, ast.Expression):
            _check(inner.body)
            return
        if isinstance(inner, ast.BoolOp):
            if not isinstance(inner.op, _ALLOWED_BOOL_OPS):
                raise ValueError("Boolean operation not allowed")
            for value in inner.values:
                _check(value)
            return
        if isinstance(inner, ast.BinOp):
            if not isinstance(inner.op, _ALLOWED_BIN_OPS):
                raise ValueError("Binary operation not allowed")
            _check(inner.left)
            _check(inner.right)
            return
        if isinstance(inner, ast.UnaryOp):
            if not isinstance(inner.op, _ALLOWED_UNARY_OPS):
                raise ValueError("Unary operation not allowed")
            _check(inner.operand)
            return
        if isinstance(inner, ast.Compare):
            for op in inner.ops:
                if not isinstance(op, _ALLOWED_COMPARE_OPS):
                    raise ValueError("Comparison not allowed")
            _check(inner.left)
            for comparator in inner.comparators:
                _check(comparator)
            return
        if isinstance(inner, ast.IfExp):
            _check(inner.test)
            _check(inner.body)
            _check(inner.orelse)
            return
        if isinstance(inner, ast.Call):
            if not isinstance(inner.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            if inner.func.id not in allowed_calls:
                raise ValueError(f"Call to '{inner.func.id}' not permitted")
            if inner.keywords:
                raise ValueError("Keyword arguments are not allowed")
            for arg in inner.args:
                _check(arg)
            return
        if isinstance(inner, ast.Name):
            if inner.id not in allowed_names:
                raise ValueError(f"Name '{inner.id}' is not allowed in expressions")
            return
        if isinstance(inner, ast.Constant):
            if isinstance(inner.value, int) and not isinstance(inner.value, bool):
                if abs(inner.value) > _MAX_NUMERIC_CONSTANT:
                    raise ValueError(f"Numeric constant too large: {inner.value}")
                return
            if isinstance(inner.value, str):
                if len(inner.value) > _MAX_STRING_LENGTH:
                    raise ValueError(f"String constant too long: {len(inner.value)} characters")
                return
            if isinstance(inner.value, (float, bool)) or inner.value is None:
                return
            raise ValueError("Unsupported constant value")
        if isinstance(inner, (ast.List, ast.Tuple)):
            for element in inner.elts:
                _check(element)
            return
        raise ValueError(
            f"Unsupported expression element: {ast.dump(inner, include_attributes=False)}"
        )

    _check(node)


def coerce_bool(value: Any) -> Optional[bool]:
    """
    Coerces common truthy and falsy representations into a boolean.

    Accepts booleans, numeric types, and strings. Numeric values are treated as truthy when nonzero. String values recognized as truthy: "true", "yes", "1", "enabled", "on"; falsy: "false", "no", "0", "disabled", "off". Whitespace and case are ignored.

    Returns:
        `True` if the value maps to a truthy representation, `False` if it maps to a falsy representation, `None` if the value cannot be determined as boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "enabled", "on"}:
            return True
        if lowered in {"false", "no", "0", "disabled", "off"}:
            return False
    return None


def to_number(value: Any) -> Optional[float]:
    """
    Convert a value to a numeric float when possible.

    Accepts ints and floats (returned as float), booleans (returned as 1.0 or 0.0), and numeric strings (commas allowed) and returns their float representation; returns None if conversion is not possible.

    Returns:
        A float representation of the input value, or `None` if the value cannot be converted to a number.
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def _safe_eval(
    expression: str,
    namespace: Mapping[str, Any],
    *,
    allowed_call_names: tuple[str, ...],
) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax") from exc
    allowed_calls = {
        name: namespace[name] for name in allowed_call_names if name in namespace
    }
    validate_safe_expression(
        tree, allowed_calls=allowed_calls, allowed_names=namespace
    )
    compiled = compile(tree, "<cli-layout-expression>", "eval")
    return eval(compiled, {"__builtins__": {}}, namespace)


def evaluate_expression(expression: str, resolve_fn: Callable[[str], Any]) -> Any:
    """
    Evaluate a layout expression using a restricted namespace that can resolve template paths.

    Parameters:
        expression (str): The expression to evaluate.
        resolve_fn (Callable[[str], Any]): Function to resolve path tokens.

    Returns:
        The result of the evaluated expression, or `None` if evaluation fails.
    """
    prepared = prepare_condition(expression)
    namespace: Dict[str, Any] = {
        "resolve": resolve_fn,
        "abs": abs,
        "min": min,
        "max": max,
    }
    try:
        return _safe_eval(
            prepared,
            namespace,
            allowed_call_names=("resolve", "abs", "min", "max"),
        )
    except (ValueError, TypeError, SyntaxError):
        return None


def evaluate_condition(expr: str, resolve_fn: Callable[[str], Any]) -> bool:
    """
    Determine whether a condition expression evaluates to true.

    The expression is prepared for evaluation (mapping token names to calls to the context resolver) and executed in a restricted namespace where `resolve` is available along with `True`, `False`, and `None`. Any evaluation error or an empty expression results in `False`.

    Parameters:
        expr (str): The condition expression to evaluate.
        resolve_fn (Callable[[str], Any]): Function to resolve tokens referenced by the expression.

    Returns:
        bool: `True` if the prepared expression evaluates truthy, `False` otherwise.
    """
    expression = expr.strip()
    if not expression:
        return False

    prepared = prepare_condition(expression)
    namespace: Dict[str, Any] = {
        "resolve": resolve_fn,
        "True": True,
        "False": False,
        "None": None,
    }
    try:
        result = _safe_eval(prepared, namespace, allowed_call_names=("resolve",))
    except (ValueError, TypeError, SyntaxError):
        return False
    return bool(result)


def prepare_condition(expr: str) -> str:
    """
    Convert a layout condition expression into a Python-evaluable expression that resolves symbols at runtime.

    The input expression may use C-style logical operators and unqualified identifiers; this function:
    - replaces `&&`/`||` with `and`/`or`,
    - normalizes boolean and null-like literals (`true`/`false`/`none`) to `True`/`False`/`None`,
    - replaces identifiers (e.g., `foo.bar`) with calls to `resolve('foo.bar')`,
    - converts unary `!` to Python `not` while preserving `!=`.

    Parameters:
        expr (str): A condition expression from the layout template.

    Returns:
        str: A Python expression string suitable for evaluation in the restricted namespace.
    """
    cleaned = expr.replace("&&", " and ").replace("||", " or ")
    tokens: list[str] = []
    index = 0
    while index < len(cleaned):
        char = cleaned[index]
        if char.isalpha() or char == "_":
            start = index
            while index < len(cleaned) and (
                cleaned[index].isalnum() or cleaned[index] in {"_", "."}
            ):
                index += 1
            token = cleaned[start:index]
            lowered = token.lower()
            if token in {"and", "or", "not", "True", "False", "None"}:
                tokens.append(token)
            elif lowered == "true":
                tokens.append("True")
            elif lowered == "false":
                tokens.append("False")
            elif lowered == "none":
                tokens.append("None")
            elif lowered in _SAFE_FUNCTION_TOKENS:
                tokens.append(token)
            else:
                tokens.append(f"resolve('{token}')")
            continue
        if char == "!":
            if index + 1 < len(cleaned) and cleaned[index + 1] == "=":
                tokens.append("!=")
                index += 2
                continue
            tokens.append("not ")
            index += 1
            continue
        tokens.append(char)
        index += 1
    return "".join(tokens)
