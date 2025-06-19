"""
Torch backend
"""

import logging

import pytest

import virtlinalg as vla
try:
    import virtlinalg.torch as vtorch
except vla.BackendNotFoundError:
    logging.warning('Could not load torch backend, tests will fail if included')

from .backend_cases import backend_cases, BackendCase

@pytest.mark.parametrize("case", backend_cases)
def test_torch(case: BackendCase) -> None:
    """
    Can perform all standard backend ops
    """
    case(vtorch)
