"""Orderflow analytics package.

Lightweight, dependency-free implementations of core orderflow metrics
used by the Meta-Scorer and microstructure veto logic.
"""
from .engine import OrderflowEngine

__all__ = ["OrderflowEngine"]
