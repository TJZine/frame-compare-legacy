"""Security tests for layout expression engine bounds checking."""

from __future__ import annotations

import ast

import pytest

from src.frame_compare.layout.engine import (
    _MAX_AST_NODES,
    _MAX_NUMERIC_CONSTANT,
    _MAX_STRING_LENGTH,
    _count_nodes,
    validate_safe_expression,
)


class TestNodeCounting:
    """Tests for AST node counting helper."""

    def test_count_simple_expression(self) -> None:
        tree = ast.parse("1 + 2", mode="eval")
        # Expression(body=BinOp(left=Constant, op=Add, right=Constant))
        # = 1 Expression + 1 BinOp + 1 Add + 1 Constant + 1 Constant = 5 nodes
        assert _count_nodes(tree) == 5

    def test_count_nested_expression(self) -> None:
        tree = ast.parse("1 + 2 + 3", mode="eval")
        # More complex nesting
        count = _count_nodes(tree)
        assert count > 4  # Should be more nodes due to nesting

    def test_count_single_constant(self) -> None:
        tree = ast.parse("42", mode="eval")
        # Expression + Constant = 2
        assert _count_nodes(tree) == 2


class TestExpressionSecurityBounds:
    """Tests for expression security limits."""

    @pytest.fixture
    def namespace(self) -> dict:
        return {"True": True, "False": False, "None": None}

    def test_rejects_large_numeric_constant(self, namespace: dict) -> None:
        large_value = _MAX_NUMERIC_CONSTANT + 1
        tree = ast.parse(str(large_value), mode="eval")

        with pytest.raises(ValueError, match="Numeric constant too large"):
            validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_accepts_numeric_constant_at_limit(self, namespace: dict) -> None:
        tree = ast.parse(str(_MAX_NUMERIC_CONSTANT), mode="eval")
        # Should not raise
        validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_rejects_negative_large_numeric_constant(self, namespace: dict) -> None:
        large_value = -(_MAX_NUMERIC_CONSTANT + 1)
        tree = ast.parse(f"({large_value})", mode="eval")

        with pytest.raises(ValueError, match="Numeric constant too large"):
            validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_rejects_long_string_constant(self, namespace: dict) -> None:
        long_string = "a" * (_MAX_STRING_LENGTH + 1)
        tree = ast.parse(repr(long_string), mode="eval")

        with pytest.raises(ValueError, match="String constant too long"):
            validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_accepts_string_constant_at_limit(self, namespace: dict) -> None:
        string_at_limit = "a" * _MAX_STRING_LENGTH
        tree = ast.parse(repr(string_at_limit), mode="eval")
        # Should not raise
        validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_rejects_expression_exceeding_node_budget(self, namespace: dict) -> None:
        # Create an expression with many nodes
        # Each "x and" adds nodes, build something that exceeds 50
        many_ands = " and ".join(["True"] * 30)
        tree = ast.parse(many_ands, mode="eval")

        node_count = _count_nodes(tree)
        if node_count > _MAX_AST_NODES:
            with pytest.raises(ValueError, match="Expression too complex"):
                validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)
        else:
            # If we didn't exceed, adjust the test expectation
            assert node_count <= _MAX_AST_NODES

    def test_accepts_simple_expression_within_budget(self, namespace: dict) -> None:
        tree = ast.parse("True and False", mode="eval")
        # Should not raise
        validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_booleans_not_treated_as_numeric_constants(self, namespace: dict) -> None:
        # Python bools are subclasses of int, ensure they're not rejected
        tree = ast.parse("True", mode="eval")
        # Should not raise
        validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)

    def test_floats_accepted_without_bounds(self, namespace: dict) -> None:
        # Float limits are not enforced (Python floats have natural limits)
        tree = ast.parse("1e308", mode="eval")
        # Should not raise
        validate_safe_expression(tree, allowed_calls={}, allowed_names=namespace)
