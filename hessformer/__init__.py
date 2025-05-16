"""HessFormer – distributed Hessian spectral analysis for Transformer models."""

from .core import estimate_spectrum, SpectralResult
__all__ = ["estimate_spectrum", "SpectralResult"]
__version__ = "0.1.0"
