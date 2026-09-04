"""
Evaluation metrics and logging utilities for Cross-Modality Re-ID.
"""

from .metrics import evaluate_cross_modal
from .logger import MetricLogger

__all__ = [
    "evaluate_cross_modal",
    "MetricLogger",
]
