"""CRISP: chemical-rule compilation into executable materials descriptors."""

from __future__ import annotations

from typing import Any

__version__ = "3.0.0"
__author__ = "Jaehwan Choi"

__all__ = [
    "compile_catalog",
]


def __getattr__(name: str) -> Any:
    """Load the API workflow only when its public name is requested."""
    if name == "compile_catalog":
        from .workflow import compile_catalog

        return compile_catalog
    raise AttributeError(name)
