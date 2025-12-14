from .adancfgd import AdaFGD, AdaNCFGD
from .snn import (
    ForwardFirstBackwardSecond,
    SNNDropout,
    SNNBatchNorm1d,
    SNNBatchNorm2d,
    SNNLinear,
    SNNLinearWithBatchNorm,
    SNNConv2d,
    SNNConv2dWithBatchNorm,
    SNN
)
from .snn_cnn import (
    pool_spikes,
    SNNCNN
)

__version__ = "0.1.4"
__author__ = "Yihe Zhu"
__email__ = "zhu.yihe@qq.com"
__description__ = "Adaptive Fractional Gradient Descent Optimizers (AdaFGD & AdaNCFGD)"
__url__ = "https://github.com/HunLuanZhiZhu/AdaNCFGD"

# Export main classes and functions
__all__ = [
    # Optimizers
    "AdaFGD",
    "AdaNCFGD",
    # SNN components
    "ForwardFirstBackwardSecond",
    "SNNDropout",
    "SNNBatchNorm1d",
    "SNNBatchNorm2d",
    "SNNLinear",
    "SNNLinearWithBatchNorm",
    "SNNConv2d",
    "SNNConv2dWithBatchNorm",
    "SNN",
    "SNNCNN",
    # Utility functions
    "pool_spikes"
]
