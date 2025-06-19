"""
Unit tests

This is a package to facilitate import of shared unit tests for backends
"""

import pytest

pytest.register_assert_rewrite("unit.backend_cases")
