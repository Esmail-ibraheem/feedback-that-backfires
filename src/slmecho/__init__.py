"""slmecho — measuring how agent harnesses condition small language models on
their own failures.

The package is deliberately dependency-light: everything outside `models` and
`scoring` is pure Python + numpy so that data generation, environments and
analysis can be exercised (and unit-tested) without loading a language model.
"""

__version__ = "0.1.0"
