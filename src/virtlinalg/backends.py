"""
Common backend utils
"""

class BackendNotFoundError(ModuleNotFoundError):
    """
    Raised by VLA backend modules when loading a backend dependency fails.

    This allows to programmatically determine if loading a VLA backend failed
    because that backend is not installed or because of a bug.
    """
