"""Evaluation code for the OntoOps experiments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ontoops")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
