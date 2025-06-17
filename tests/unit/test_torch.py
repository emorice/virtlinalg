"""
Torch backend
"""

import logging

import pytest

from virtlinalg import BackendNotFoundError
try:
    import virtlinalg.torch as vtorch
except BackendNotFoundError:
    logging.warning('Could not load torch backend, tests will fail if included')

from .backend_cases import backend_cases

@pytest.mark.parametrize("case", backend_cases)
def test_torch(case):
    """
    Can perform all standard backend ops
    """
    case(vtorch)
