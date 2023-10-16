"""Package imports"""
import warnings
from .discscsp import *
from .affscsp import *
try:
    import torch
    from .torchscsp import *
except ImportError:
    warnings.warn(
      "PyTorch is not installed. Therefore torchscsp cannot be imported."
      "Please install the torch package to use it."
    )
