# This file contains intentional type errors to verify that type stubs work.
# CI runs type checkers on this file and expects them to fail.
# Do not import this file in actual tests.

import nanobind_example as m


def bad_add_call(a: str, b: str) -> int:
    """Intentionally passing str to int parameters - should fail type checking."""
    return m.add(a, b)
