"""
Data acquisition, preprocessing, synthetic IR generation, and dataset management module.
"""

from .synthesize_ir import generate_synthetic_ir, synthesize_dataset
from .dataset import BaseCrossModalDataset, get_cross_modal_dataloaders
from .prepare_sysu_regdb import SYSUPrepare, RegDBPrepare

__all__ = [
    "generate_synthetic_ir",
    "synthesize_dataset",
    "BaseCrossModalDataset",
    "get_cross_modal_dataloaders",
    "SYSUPrepare",
    "RegDBPrepare",
]
