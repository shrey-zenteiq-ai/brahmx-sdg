"""
Symbolic / Numeric Validator.

Checks formulas, units, numerical consistency, and counterfactual arithmetic.
Science tasks need more than linguistic fluency — this catches unit mismatches,
impossible values, failed balances, and solver-inconsistent reasoning.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import structlog

logger = structlog.get_logger()


@dataclass
class SymbolicCheckResult:
    check_id: str
    check_type: str  # symbolic, numeric, unit_consistency, range
    passed: bool
    expression: str = ""
    expected: str = ""
    actual: str = ""
    explanation: str = ""


class SymbolicNumericValidator:
    """Validates mathematical expressions, units, and numerical claims."""

    def validate_all(self, checks: list[dict[str, Any]]) -> list[SymbolicCheckResult]:
        results = []
        for check in checks:
            result = self._dispatch(check)
            results.append(result)
        return results

    def _dispatch(self, check: dict) -> SymbolicCheckResult:
        check_type = check.get("check_type", "symbolic")
        handlers = {
            "symbolic": self._check_symbolic,
            "numeric": self._check_numeric,
            "unit_consistency": self._check_units,
            "range": self._check_range,
        }
        handler = handlers.get(check_type, self._check_symbolic)
        return handler(check)

    def _check_symbolic(self, check: dict) -> SymbolicCheckResult:
        """Verify symbolic expression equality using SymPy."""
        try:
            from sympy import simplify, sympify
            expr = check.get("expression", "")
            expected = check.get("expected", "")
            lhs = sympify(expr)
            rhs = sympify(expected)
            passed = simplify(lhs - rhs) == 0
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="symbolic",
                passed=passed,
                expression=expr,
                expected=expected,
                actual=str(lhs),
            )
        except Exception as e:
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="symbolic",
                passed=False,
                explanation=f"Parse error: {e}",
            )

    def _check_numeric(self, check: dict) -> SymbolicCheckResult:
        """Verify numerical computation within tolerance."""
        try:
            from sympy import N, sympify
            expr = sympify(check.get("expression", "0"))
            expected = float(check.get("expected", "0"))
            tolerance = float(check.get("tolerance", 1e-6))
            actual = float(N(expr))
            passed = abs(actual - expected) <= tolerance
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="numeric",
                passed=passed,
                expression=str(expr),
                expected=str(expected),
                actual=str(actual),
            )
        except Exception as e:
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="numeric",
                passed=False,
                explanation=f"Evaluation error: {e}",
            )

    def _check_units(self, check: dict) -> SymbolicCheckResult:
        """Verify unit consistency using Pint."""
        try:
            import pint
            ureg = pint.UnitRegistry()
            from_val = ureg.Quantity(check.get("from_value", "1 meter"))
            to_val = ureg.Quantity(check.get("to_value", "1 meter"))
            passed = from_val.dimensionality == to_val.dimensionality
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="unit_consistency",
                passed=passed,
                expression=str(from_val),
                expected=str(to_val.dimensionality),
                actual=str(from_val.dimensionality),
            )
        except Exception as e:
            return SymbolicCheckResult(
                check_id=check.get("check_id", ""),
                check_type="unit_consistency",
                passed=False,
                explanation=f"Unit error: {e}",
            )

    def _check_range(self, check: dict) -> SymbolicCheckResult:
        """Verify a value falls within an expected range."""
        value = float(check.get("value", 0))
        low = float(check.get("min", float("-inf")))
        high = float(check.get("max", float("inf")))
        passed = low <= value <= high
        return SymbolicCheckResult(
            check_id=check.get("check_id", ""),
            check_type="range",
            passed=passed,
            expression=str(value),
            expected=f"[{low}, {high}]",
            actual=str(value),
        )
