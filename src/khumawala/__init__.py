"""
Khumawala Rules Implementation for Uncapacitated Facility Location Problems

This package contains implementations of various Khumawala rules and branching
strategies for solving uncapacitated facility location problems using 
pseudo-Boolean polynomial representations.
"""

from .analysis import TermAnalyzer
from .rules import KhumawalaRules

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    "TermAnalyzer",
    "KhumawalaRules"
] 